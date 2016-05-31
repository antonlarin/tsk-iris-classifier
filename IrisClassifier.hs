import System.IO
import System.Random (randomRIO, mkStdGen, setStdGen)
import Data.Array.IO (IOArray, newListArray, readArray, writeArray)
import Control.Monad (forM)
import Data.Char (toLower)
import Data.List (unzip5, zip5, sortBy)
import qualified Data.Sequence as Sqnc
import Data.Foldable (toList, foldl')
import Data.Function (on)

main :: IO ()
main = do
    input <- openFile "iris.data" ReadMode
    dataset <- readDataset input
    hClose input
    let seed = 154
    setStdGen $ mkStdGen seed
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
irisFromDouble val | 0 <= val && val < 1 = Setosa
                   | 1 <= val && val < 2 = Versicolor
                   | 2 <= val && val <= 3 = Virginica


type DataItem = (Double, Double, Double, Double, Iris)
type DataSet = [DataItem]
itemClass :: DataItem -> Iris
itemClass (_, _, _, _, iris) = iris

type ProcessedDataItem = (Double, Double, Double, Double, Double)

procItemResponse :: ProcessedDataItem -> Double
procItemResponse (_, _, _, _, resp) = resp

procItemFeatures :: ProcessedDataItem -> (Double, Double, Double, Double)
procItemFeatures (a, b, c, d, _) = (a, b, c, d)


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

antecedent :: ([Double], [Double], [Double]) -> Double
antecedent (xs, as, cs) = product $ map antecedentConjunct $ zip3 xs as cs
    where antecedentConjunct (x, a, c) = exp ((x - c) ** 2 / ((-2) * a * a))

predict :: Model -> (Double, Double, Double, Double) -> Double
predict (TSKZero rulesCount aSets cSets bs) (x, y, z, w) =
    let inputSets = replicate rulesCount [x, y, z, w]
        antecedents = map antecedent $ zip3 inputSets aSets cSets
        numerator = sum $ map tupleProduct $ zip antecedents bs
        denominator = sum antecedents
    in numerator / denominator
    where tupleProduct (p, b) = p * b

buildAndTestModel :: (DataSet, DataSet) -> IO (Double, Double)
buildAndTestModel (train, test) = do
        let procTrain = preprocess train
        let procTest = preprocess test
        let testClasses = map itemClass test
        model <- identifyModel procTrain
        let preOptPredictions = map (predictClass model) procTest
        let preOptScore = computeScore preOptPredictions testClasses
        model <- optimizeModel model procTrain
        let postOptPredictions = map (predictClass model) procTest
        let postOptScore = computeScore postOptPredictions testClasses
        return (preOptScore, postOptScore)
    where predictClass m = irisFromDouble . predict m . procItemFeatures

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
euclidean (x:xs) (y:ys) = (x - y) ** 2 + euclidean xs ys

d :: [Double] -> [Double] -> Int -> Int -> Double
d x c n nSum = (fromIntegral n) * euclidean c x / (fromIntegral nSum)

newCluster c x alpha = map updateClusterElems (zip c x)
    where updateClusterElems (ce, xe) = ce + alpha * (xe - ce)

structIdEpoch :: Int -> Int -> Sqnc.Seq [Double] -> Sqnc.Seq Int ->
    Double -> Double -> [ProcessedDataItem] -> Double -> Sqnc.Seq [Double]
structIdEpoch epoch maxEpoch cs _ _ _ _ _| epoch > maxEpoch = cs
structIdEpoch epoch maxEpoch cs ns alphaW alphaR ds eps =
    let epochRatio = (fromIntegral epoch) / (fromIntegral maxEpoch)
        iteration (cs, ns, alphaW, alphaR) (x, z, u, v, _) =
            let ftrs = [x, z, u, v]
                indices = Sqnc.fromFunction (length cs) id
                nSum = sum ns
                criterion = compare `on` \(_, cs, n) -> d ftrs cs n nSum
                clusters = Sqnc.zip3 indices cs ns
                [(wi, w, wn), (ri, r, rn)] = toList $ Sqnc.take 2
                        (Sqnc.sortBy criterion clusters)
                newNs = Sqnc.update wi (wn + 1) ns
                newCs = Sqnc.update wi (newCluster w ftrs alphaW) $
                        Sqnc.update ri (newCluster r ftrs (-alphaR)) cs
                newAlphaW = alphaW * (1.0 - epochRatio)
                newAlphaR = alphaR * (1.0 - epochRatio)
            in (newCs, newNs, newAlphaW, newAlphaR)
        (newCs, newNs, newAlphaW, newAlphaR) = foldl' iteration (cs, ns, alphaW, alphaR) ds
        diff = avg (\(oldC, newC, newN) -> d oldC newC newN (sum newNs)) $
                toList (Sqnc.zip3 cs newCs newNs)
    in if diff < eps
       then newCs
       else structIdEpoch (epoch + 1) maxEpoch newCs newNs newAlphaW newAlphaR ds eps

identifyModel :: [ProcessedDataItem] -> IO Model
identifyModel dataset = do
        let startNs = Sqnc.replicate clusterCountLimit 1
        -- produce initial distribution for c
        let startCs = Sqnc.replicate clusterCountLimit [0.5, 0.5, 0.5, 0.5]
        let cs = structIdEpoch 1 maxEpoch startCs startNs alphaW alphaR dataset epsilon
        let filteredCs = toList (Sqnc.filter (all insideZeroOne) cs)
        let as = toList (map (findAs filteredCs) filteredCs)
        let features = map procItemFeatures dataset
        let responses = map procItemResponse dataset
        let bs = map (findB features responses) (zip as filteredCs)
        return (TSKZero (length as) as filteredCs bs)
    where
        clusterCountLimit = 20
        maxEpoch = 5
        epsilon = 0.0001
        alphaW = 0.7
        alphaR = 0.5
        insideZeroOne v = 0 <= v && v <= 1
        r = 1.5
        findAs cs c = let (ck:ch:_) = sortBy (closestToC) cs
                      in replicate 4 (euclidean ck ch / r)
            where closestToC = compare `on` (\c' -> euclidean c c')
        findB ftrs rspns (as, cs) = if denom == 0
                                    then 0.0
                                    else numer / denom
            where step (numer', denom') ((x, z, u, v), y) =
                        let alpha = antecedent ([x, z, u, v], as, cs)
                        in (numer' + alpha * y, denom' + alpha)
                  (numer, denom) = foldl' step (0.0, 0.0) $ zip ftrs rspns

optimizeModel :: Model -> [ProcessedDataItem] -> IO Model
optimizeModel m _ = return m

