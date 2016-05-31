import System.IO
import System.Random (randomRIO, mkStdGen, setStdGen)
import Data.Array.IO (IOArray, newListArray, readArray, writeArray)
import Control.Monad (forM)
import Data.Char (toLower)
import Data.List (unzip5, zip5, zip6, sortBy)
import qualified Data.Sequence as Sqnc
import Data.Foldable (toList, foldl')
import Data.Function (on)

main :: IO ()
main = do
    input <- openFile "iris.data" ReadMode
    dataset <- readDataset input
    hClose input
    let seed = 154
    --setStdGen $ mkStdGen seed
    shuffledDataset <- shuffle dataset
    let folds = 10
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

insideZeroOne :: Double -> Bool
insideZeroOne v = 0 <= v && v <= 1

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

paramsVector :: Model -> ([Double], Int, Int)
paramsVector (TSKZero clusterCount as cs bs) =
        (flatten as ++ flatten cs ++ bs, clusterCount, length (head as))
    where flatten xss = foldl' (++) [] xss

fromParamsVector :: ([Double], Int, Int) -> Model
fromParamsVector (xs, clusterCount, featureCount) =
    let
        (as, xs') = extractACs clusterCount xs
        (cs, bs) = extractACs clusterCount xs'
    in TSKZero clusterCount as cs bs
    where
        extractACs clustersLeft xs | clustersLeft == 0 = ([], xs)
        extractACs clustersLeft xs = (cluster : otherACs, otherVec)
            where (cluster, rest) = splitAt featureCount xs
                  (otherACs, otherVec) = extractACs (clustersLeft - 1) rest

euclidean :: [Double] -> [Double] -> Double
euclidean [] [] = 0.0
euclidean xs ys | length xs /= length ys = error "Euclidean: uneven lists"
euclidean (x:xs) (y:ys) = (x - y) ** 2 + euclidean xs ys

d :: [Double] -> [Double] -> Int -> Int -> Double
d x c n nSum = (fromIntegral n) * euclidean c x / (fromIntegral nSum)

newCluster c x alpha = map updateClusterElems (zip c x)
    where updateClusterElems (ce, xe) = ce + alpha * (xe - ce)

structIdEpoch :: Int -> Int -> Sqnc.Seq [Double] -> Sqnc.Seq Int ->
    Double -> Double -> [ProcessedDataItem] -> Double -> (Sqnc.Seq [Double], Int, Double)
structIdEpoch epoch maxEpoch cs _ _ _ _ _| epoch > maxEpoch = (cs, epoch - 1, 0.0)
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
       then (newCs, epoch, diff)
       else structIdEpoch (epoch + 1) maxEpoch newCs newNs newAlphaW newAlphaR ds eps

identifyModel :: [ProcessedDataItem] -> IO Model
identifyModel dataset = do
        let startNs = Sqnc.replicate clusterCountLimit 1
        let rand = \_ -> randomRIO (0.0, 1.0)
        startCs <- forM (Sqnc.fromList [1..clusterCountLimit]) $
                   \_ -> forM [1..4] rand
        let (cs, epochs, diff) = structIdEpoch 1 maxEpoch startCs startNs alphaW alphaR dataset epsilon
        putStrLn $ "Epochs required: " ++ show epochs
        putStrLn $ "Diff achieved: " ++ show diff
        let filteredCs = toList (Sqnc.filter (all insideZeroOne) cs)
        let as = toList (map (findAs filteredCs) filteredCs)
        let features = map procItemFeatures dataset
        let responses = map procItemResponse dataset
        let bs = map (findB features responses) (zip as filteredCs)
        return (TSKZero (length as) as filteredCs bs)
    where
        clusterCountLimit = 30
        maxEpoch = 10
        epsilon = 0.0001
        alphaW = 0.06
        alphaR = 0.02
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


costFunction :: Int -> Int -> [ProcessedDataItem] -> [Double] -> Double
costFunction clusterCount featureCount dataset xs =
    let model = fromParamsVector (xs, clusterCount, featureCount)
        localCost (x, z, u, v, y) = (predict model (x, z, u, v) - y) ** 2 
    in avg id (map localCost dataset)

constraints :: Int -> Int -> [Double] -> [Bool]
constraints clusterCount featureCount xs =
        checkA (clusterCount * featureCount) xs
    where
        checkA 0 ys = checkC (clusterCount * featureCount) ys
        checkA elemsLeft (y:ys) = (y > 0) : checkA (elemsLeft - 1) ys

        checkC 0 ys = checkB clusterCount ys
        checkC elemsLeft (y:ys) = insideZeroOne y : checkC (elemsLeft - 1) ys

        checkB 0 [] = []
        checkB elemsLeft (y:ys) = someClass y : checkB (elemsLeft - 1) ys
            where someClass y = 0.0 <= y && y <= 3.0


psoIteration :: Int -> Int -> Double ->
    [[Double]] -> [[Double]] -> [[Double]] -> [Double] ->
    ([Double] -> Double) -> ([Double] -> [Bool]) -> IO [[Double]]
psoIteration iter maxIter _  xs _ _ _ _ _ | iter > maxIter = return xs
psoIteration iter maxIter costThreshold xs vs ps pg cost checkConstraints =
    let
    in do
        let components = length xs
        rands1 <- forM [1..components] $ \_ -> randomRIO (0.0, 1.0)
        rands2 <- forM [1..components] $ \_ -> randomRIO (0.0, 1.0)
        let newVs = map stepV $ zip5 xs vs ps rands1 rands2
        let (newXs, correctedVs) = unzip $ map stepX $ zip xs newVs
        let newPs = map chooseLessCostly $ zip ps xs
        let newPg = head $ sortBy (compare `on` cost) (pg : xs)
        if cost newPg < costThreshold
        then return newXs
        else psoIteration (iter + 1) maxIter costThreshold
                    newXs correctedVs newPs newPg cost checkConstraints
    where
        w = 0.5
        c1 = 0.3
        c2 = 0.6

        stepV (x, v, p, rand1, rand2) = map stepVComponent $
                zip6 x v p pg replicatedRand1 replicatedRand2
            where stepVComponent (xe, ve, pe, pge, rand1, rand2) = w * ve +
                    c1 * rand1 * (pe - xe) + c2 * rand2 * (pge - xe)
                  replicatedRand1 = replicate (length x) rand1
                  replicatedRand2 = replicate (length x) rand2

        stepX (x, v) =
            let newX = map (\(xe, ve) -> xe + ve) $ zip x v -- no constraints
                brokenConstraints = checkConstraints newX
            in if any id brokenConstraints
               then stepX (x, shrinkV v brokenConstraints)
               else (x, v)
            where shrinkV v constraintsMask =
                    map (\(v, p) -> if p then 0.5 * v else v) $
                    zip v constraintsMask

        chooseLessCostly (a, b) | cost a < cost b = a
                                | otherwise       = b

optimizeModel :: Model -> [ProcessedDataItem] -> IO Model
optimizeModel model dataset = do
        let (x0, clusterCount, featureCount) = paramsVector model
        let n = length x0
        let rand = \_ -> randomRIO (0.0, 1.0)
        otherXs <- forM [1..(swarmSize - 1)] $
                   \_ -> forM [1..n] rand
        let xs = x0 : otherXs
        let vs = replicate swarmSize (replicate n 0.0)
        let cost = costFunction clusterCount featureCount dataset
        let pg = head (sortBy (compare `on` cost) xs)

        optimizedXs <- psoIteration 1 maxIterations costThreshold xs vs xs pg cost (constraints clusterCount featureCount)
        let optimizedParams = head optimizedXs
        return (fromParamsVector (optimizedParams, clusterCount, featureCount))
    where
        swarmSize = 50
        maxIterations = 50
        costThreshold = 0.1

