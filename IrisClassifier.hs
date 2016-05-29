import System.IO
import System.Random (randomRIO)
import Data.Array.IO (IOArray, newListArray, readArray, writeArray)
import Control.Monad (forM)
import Data.Char (toLower)
import Data.List (unzip5, zip5)
import qualified Data.Sequence as Sqnc
import Data.Foldable (toList, foldl')
import Data.Function (on)

main :: IO ()
main = do
    input <- openFile "iris.data" ReadMode
    dataset <- readDataset input
    hClose input
    shuffledDataset <- shuffle dataset
    let folds = 5
    let trainTestPairs = crossValidationSplit folds shuffledDataset
    scores <- mapM buildAndTestModel trainTestPairs
    let (preOptScores, postOptScores) = unzip scores
    let meanPreOptScore = sum preOptScores / (fromIntegral folds)
    let meanPostOptScore = sum postOptScores / (fromIntegral folds)
    putStrLn $ "Pre opt: " ++ show meanPreOptScore
    putStrLn $ "Post opt: " ++ show meanPostOptScore


-- utility definitions
shuffle :: [a] -> IO [a]
shuffle xs = do
        arr <- listToArray n xs
        forM [1..n] $ \i -> do
            j <- randomRIO (i, n)
            xi <- readArray arr i
            xj <- readArray arr j
            writeArray arr j xi
            return xj
    where n = length xs
          listToArray :: Int -> [a] -> IO (IOArray Int a)
          listToArray n xs = newListArray (1, n) xs

splitWith :: Char -> String -> [String]
splitWith _ [] = []
splitWith sep lines = front : splitWith sep back
            where notSep = (/= sep)
                  front = takeWhile notSep lines
                  backWithSep = dropWhile notSep lines
                  back = if null backWithSep
                         then []
                         else tail backWithSep

extractFolds :: [a] -> [Int] -> [([a], [a])]
extractFolds xs lens = extractFolds' xs [] lens
    where extractFolds' :: [a] -> [a] -> [Int] -> [([a], [a])]
          extractFolds' _ _ [] = []
          extractFolds' xs front (len:lens) =
                        (newFold, front ++ rest) :
                        extractFolds' rest (front ++ newFold) lens
                    where (newFold, rest) = splitAt len xs

avg :: Fractional b => (a -> b) -> [a] -> b
avg f xs = (foldl' (+) 0.0 (map f xs)) / (fromIntegral (length xs))


-- problem-specific definitions
data Iris = Setosa
          | Versicolor
          | Virginica
          deriving (Show, Eq)

irisFromString :: String -> Iris
irisFromString s = case s of
    "Iris-setosa" -> Setosa
    "Iris-versicolor" -> Versicolor
    "Iris-virginica" -> Virginica
    otherwise -> error "Can't parse iris class"

irisToString :: Iris -> String
irisToString iris = "Iris-" ++ decapitalize (show iris)
            where decapitalize "" = ""
                  decapitalize (c:cs) = toLower c : cs

irisToDouble :: Iris -> Double
irisToDouble Setosa = 0.5
irisToDouble Versicolor = 1.5
irisToDouble Virginica = 2.5

irisFromDouble :: Double -> Iris
irisFromDouble val | val < 1 = Setosa
irisFromDouble val | val >= 1 && val < 2 = Versicolor
irisFromDouble val | val >= 1 && val <= 3 = Virginica


type DataItem = (Double, Double, Double, Double, Iris)
type DataSet = [DataItem]
itemClass :: DataItem -> Iris
itemClass (_, _, _, _, iris) = iris

itemFeatures :: DataItem -> (Double, Double, Double, Double)
itemFeatures (a, b, c, d, _) = (a, b, c, d)

type ProcessedDataItem = (Double, Double, Double, Double, Double)


readDataset :: Handle -> IO DataSet
readDataset input = readDataset' input []

readDataset' :: Handle -> DataSet -> IO DataSet
readDataset' input xs = do
    endOfInput <- hIsEOF input
    if endOfInput
    then return xs
    else do
        line <- hGetLine input
        let entries = splitWith ',' line
        if length entries == 5
        then do
            let toDouble = \n -> read n :: Double
            let [a, b, c, d] = map toDouble $ take 4 entries
            let entryClass = head $ drop 4 entries
            let dataItem = (a, b, c, d, irisFromString entryClass)
            readDataset' input (dataItem : xs)
        else do
            readDataset' input xs

preprocess :: DataSet -> [ProcessedDataItem]
preprocess dataset = zip5 as' bs' cs' ds' irises'
            where (as, bs, cs, ds, irises) = unzip5 dataset
                  irises' = map irisToDouble irises
                  [as', bs', cs', ds'] = map normalize [as, bs, cs, ds]
                  normalize xs = let min = minimum xs
                                     max = maximum xs
                                     invSpan = 1.0 / (max - min)
                                 in map (\x -> invSpan * (x - min)) xs 

computeScore :: [Iris] -> [Iris] -> Double
computeScore predictions answers | null predictions || null answers = 0.0
computeScore predictions answers =
    let denominator = length answers
        numerator = length $ filter id $ zipWith (==) predictions answers
    in fromIntegral numerator / fromIntegral denominator

predict :: Model -> (Double, Double, Double, Double) -> Double
predict (TSKZero rulesCount aSets cSets bs) (x, y, z, w) =
    let antecedent (x, a, c) = exp $ (x - c) ** 2 / ((-2) * a * a)
        aggregate (as, cs, xs) = product $ map antecedent $ zip3 xs as cs
        inputSets = replicate rulesCount [x, y, z, w]
        rules = map aggregate $ zip3 aSets cSets inputSets
        numerator = sum $ map (\(p, b) -> p * b) $ zip rules bs
        denominator = sum rules
    in numerator / denominator

buildAndTestModel :: (DataSet, DataSet) -> IO (Double, Double)
buildAndTestModel (train, test) = do
        let procTrain = preprocess train
        let procTest = preprocess test
        let testClasses = map itemClass test
        model <- identifyModel procTrain
        let preOptPredictions = map (predictClass model) test
        let preOptScore = computeScore preOptPredictions testClasses
        model <- optimizeModel model procTrain
        let postOptPredictions = map (predictClass model) test
        let postOptScore = computeScore postOptPredictions testClasses
        return (preOptScore, postOptScore)
    where predictClass m = irisFromDouble . predict m . itemFeatures

crossValidationSplit :: Int -> DataSet -> [(DataSet, DataSet)]
crossValidationSplit folds dataset =
        let size = length dataset
            smallerFoldSize = size `div` folds
            leftover = size `mod` folds
            biggerFoldSize = smallerFoldSize + 1
            foldSizes = replicate leftover smallerFoldSize ++
                        replicate (folds - leftover) biggerFoldSize
        in extractFolds dataset foldSizes


-- Model-specific definitions
data Model = TSKZero Int [[Double]] [[Double]] [Double]

euclidean :: [Double] -> [Double] -> Double
euclidean [] [] = 0.0
euclidean xs ys | length xs /= length ys = error "Euclidean: uneven lists"
euclidean (x:xs) (y:ys) = abs (x - y) ** 2 + euclidean xs ys

d :: [Double] -> [Double] -> Int -> Sqnc.Seq Int -> Double
d x c n ns = (fromIntegral n) * euclidean c x / (fromIntegral $ sum ns)

newCluster c x alpha = map updateClusterElems cWithX
    where updateClusterElems (ce, xe) = ce + alpha * (xe - ce)
          cWithX = zip c x

structIdEpoch :: Int -> Int -> Sqnc.Seq [Double] -> Sqnc.Seq Int ->
    Double -> Double -> [ProcessedDataItem] -> Double -> Sqnc.Seq [Double]
structIdEpoch epoch maxEpoch cs _ _ _ _ _| epoch == maxEpoch = cs
structIdEpoch epoch maxEpoch cs ns alphaW alphaR ds eps =
    let iteration (cs, ns, alphaW, alphaR) (x, y, z, v, _) =
            let ftrs = [x, y, z, v]
                indices = Sqnc.fromFunction (length cs) id
                criterion = compare `on` \(_, cs, n) -> d ftrs cs n ns
                clusters = Sqnc.zip3 indices cs ns
                [(wi, w, wn), (ri, r, rn)] = toList $ Sqnc.take 2
                        (Sqnc.sortBy criterion clusters)
                newNs = Sqnc.update wi (wn + 1) ns
                mAlphaR = (-1) * alphaR
                newCs = (Sqnc.update wi (newCluster w ftrs alphaW)) $
                        (Sqnc.update ri (newCluster r ftrs malphaR)) cs
                epochRatio = (fromIntegral epoch) / (fromIntegral maxEpoch)
                newAlphaW = alphaW * (1.0 - epochRatio)
                newAlphaR = alphaR * (1.0 - epochRatio)
            in (newCs, newNs, newAlphaW, newAlphaR)
        (newCs, newNs, newAlphaW, newAlphaR) = foldl' iteration (cs, ns, alphaW, alphaR) ds
        diff = avg (\(oldC, newC, newN) -> d oldC newC newN newNs) $
                toList (Sqnc.zip3 cs newCs newNs)
    in if diff < eps
       then newCs
       else structIdEpoch (epoch + 1) maxEpoch newCs newNs newAlphaW newAlphaR ds eps

identifyModel :: [ProcessedDataItem] -> IO Model
-- identifyModel dataset = do
        -- let clusterWins = replicate clusterCountLimit 1
    -- where
        -- clusterCountLimit = 20
        -- maxEpoch = 5
        -- epsilon = 0.0001
        -- alphaW = 0.7
        -- alphaR = 0.5
    
identifyModel dataset = return (TSKZero 0 [] [] [])

optimizeModel :: Model -> [ProcessedDataItem] -> IO Model
optimizeModel m _ = return m

