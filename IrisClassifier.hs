import System.IO
import System.Random (randomRIO, mkStdGen, setStdGen)
import Data.Array.IO (IOArray, newListArray, readArray, writeArray)
import Control.Monad (forM)
import Data.Char (toLower)
import Data.List (zip4, unzip5, zip5, sortBy, sort)
import qualified Data.Sequence as Sqnc
import Data.Foldable (toList, foldl', minimumBy)
import Data.Function (on)
import Debug.Trace

main :: IO ()
main = do
    input <- openFile "iris.data" ReadMode
    dataset <- readDataset input
    hClose input
    let seed = 123
    setStdGen $ mkStdGen seed
    shuffledDataset <- shuffle dataset
    let processedDataset = preprocess shuffledDataset
    let folds = 10
    let trainTestPairs = crossValidationSplit folds processedDataset
    scores <- mapM buildAndTestModel trainTestPairs
    let (preOptScores, postOptScores) = unzip scores
    let meanPreOptScore = avg preOptScores
    let meanPostOptScore = avg postOptScores
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

splitWith :: (a -> Bool) -> [a] -> [[a]]
splitWith _ [] = []
splitWith isSep xs = front : splitWith isSep back
    where (front, back) = splitInTwo ([], xs)
          splitInTwo res@(_, xs) | null xs = res
          splitInTwo (partialFront, xs@(x:xs')) =
                if isSep x
                then (partialFront, xs')
                else splitInTwo (partialFront ++ [x], xs')

extractFolds :: [a] -> [Int] -> [([a], [a])]
extractFolds xs lens = extractFolds' xs [] lens
    where extractFolds' :: [a] -> [a] -> [Int] -> [([a], [a])]
          extractFolds' _ _ [] = []
          extractFolds' xs front (len:lens) =
                        (front ++ rest, newFold) :
                        extractFolds' rest (front ++ newFold) lens
                    where (newFold, rest) = splitAt len xs

avgBy :: Fractional b => (a -> b) -> [a] -> b
avgBy _ xs | null xs = 0.0
avgBy f xs = (foldl' (+) 0.0 (map f xs)) / (fromIntegral (length xs))

avg :: Fractional a => [a] -> a
avg = avgBy id

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
                   | otherwise = error "Can't convert Double to Iris"


type DataItem = (Double, Double, Double, Double, Iris)
type Dataset = [DataItem]
itemClass :: DataItem -> Iris
itemClass (_, _, _, _, iris) = iris

type ProcessedDataItem = (Double, Double, Double, Double, Double)
type ProcessedDataset = [ProcessedDataItem]

procItemResponse :: ProcessedDataItem -> Double
procItemResponse (_, _, _, _, resp) = resp

procItemFeatures :: ProcessedDataItem -> (Double, Double, Double, Double)
procItemFeatures (a, b, c, d, _) = (a, b, c, d)


readDataset :: Handle -> IO Dataset
readDataset input = readDataset' input []

readDataset' :: Handle -> Dataset -> IO Dataset
readDataset' input xs = do
    endOfInput <- hIsEOF input
    if endOfInput
    then return xs
    else do
        line <- hGetLine input
        let entries = splitWith (== ',') line
        if length entries == 5
        then do
            let toDouble = \n -> read n :: Double
            let [a, b, c, d] = map toDouble $ take 4 entries
            let entryClass = head $ drop 4 entries
            let dataItem = (a, b, c, d, irisFromString entryClass)
            readDataset' input (dataItem : xs)
        else readDataset' input xs

preprocess :: Dataset -> ProcessedDataset
preprocess dataset = zip5 xs' zs' us' vs' irises'
            where (xs, zs, us, vs, irises) = unzip5 dataset
                  irises' = map irisToDouble irises
                  [xs', zs', us', vs'] = map normalize [xs, zs, us, vs]
                  normalize ws = let min = minimum ws
                                     max = maximum ws
                                     invSpan = 1.0 / (max - min)
                                 in map (\x -> invSpan * (x - min)) ws 

computeScore :: [Iris] -> [Iris] -> Double
computeScore predictions answers = avgBy matchAsInt $ zip predictions answers
    where matchAsInt (a, b) = if a == b then 1.0 else 0.0

antecedent :: ([Double], [Double], [Double]) -> Double
antecedent (xs, as, cs) = product $ map antecedentConjunct $ zip3 xs as cs
    where antecedentConjunct (x, a, c) = exp ((x - c) ** 2 / (-2 * a * a))

predict :: Model -> (Double, Double, Double, Double) -> Double
predict (TSKZero rulesCount aSets cSets bs) (x, y, z, w) =
    let xSets = replicate rulesCount [x, y, z, w]
        antecedents = map antecedent $ zip3 xSets aSets cSets
        numerator = sum $ map tupleProduct $ zip antecedents bs
        denominator = sum antecedents
    in numerator / denominator
    where tupleProduct (p, b) = p * b

buildAndTestModel :: (ProcessedDataset, ProcessedDataset) ->
        IO (Double, Double)
buildAndTestModel (train, test) =
    putStrLn ("Train: " ++ (show $ length train) ++ ", test: " ++
        (show $ length test)) >>
    identifyModel train >>= \model ->
    optimizeModel model train >>= \model' ->
    let testClasses = map (irisFromDouble . procItemResponse) test
        predictClass m = irisFromDouble . predict m . procItemFeatures
        preOptPredictions = map (predictClass model) test
        preOptScore = computeScore preOptPredictions testClasses
        postOptPredictions = map (predictClass model') test
        postOptScore = computeScore postOptPredictions testClasses
    in return (preOptScore, postOptScore)

crossValidationSplit :: Int -> ProcessedDataset ->
        [(ProcessedDataset, ProcessedDataset)]
crossValidationSplit folds dataset =
        let size = length dataset
            smallerFoldSize = size `div` folds
            leftover = size `mod` folds
            biggerFoldSize = smallerFoldSize + 1
            foldSizes = replicate leftover biggerFoldSize ++
                        replicate (folds - leftover) smallerFoldSize
        in extractFolds dataset foldSizes


-- Model-specific definitions
data Model = TSKZero Int [[Double]] [[Double]] [Double]
             deriving Show

paramsVector :: Model -> ([Double], Int, Int)
paramsVector (TSKZero clusterCount as cs bs) =
        (flatten as ++ flatten cs ++ bs, clusterCount, length (head as))
    where flatten xss = foldr (++) [] xss

fromParamsVector :: ([Double], Int, Int) -> Model
fromParamsVector (xs, clusterCount, featureCount) =
    let
        (as, xs') = unflatten clusterCount xs
        (cs, bs) = unflatten clusterCount xs'
    in TSKZero clusterCount as cs bs
    where
        unflatten 0 xs = ([], xs)
        unflatten clustersLeft xs = (cluster : otherACs, otherVec)
            where (cluster, rest) = splitAt featureCount xs
                  (otherACs, otherVec) = unflatten (clustersLeft - 1) rest

euclidean :: [Double] -> [Double] -> Double
euclidean xs ys | length xs /= length ys = error "Euclidean: uneven lists"
                | otherwise = foldl' step 0.0 $ zip xs ys
    where step acc (x, y) = acc + (x - y) ** 2

sqrtEuclidean :: [Double] -> [Double] -> Double
sqrtEuclidean xs ys = sqrt $ euclidean xs ys

d :: [Double] -> [Double] -> Int -> Int -> Double
d x c n nSum = (fromIntegral n) * euclidean c x / (fromIntegral nSum)

updateCluster :: [Double] -> [Double] -> Double -> [Double]
updateCluster c x alpha = map updateClusterElem (zip c x)
    where updateClusterElem (ce, xe) = ce + alpha * (xe - ce)

structIdIteration :: Double -> Double ->
        (Sqnc.Seq [Double], Sqnc.Seq Int) -> ProcessedDataItem ->
        (Sqnc.Seq [Double], Sqnc.Seq Int)
structIdIteration alphaW alphaR (cs, ns) (x, z, u, v, _) =
    let ftrs = [x, z, u, v]
        indices = Sqnc.fromList [0..(length cs - 1)]
        nSum = sum ns
        criterion = compare `on` \(_, c, n) -> d ftrs c n nSum
        clusters = Sqnc.zip3 indices cs ns
        ((wi, w, wn):(ri, r, _):_) = toList $
                Sqnc.unstableSortBy criterion clusters
        ns' = Sqnc.update wi (wn + 1) ns
        cs' = Sqnc.update wi (updateCluster w ftrs alphaW) $
                Sqnc.update ri (updateCluster r ftrs (-alphaR)) cs
    in (cs', ns')

structIdEpoch :: Int -> Int -> Sqnc.Seq [Double] -> Sqnc.Seq Int ->
    Double -> Double -> [ProcessedDataItem] -> Double -> (Sqnc.Seq [Double], Int, Double)
structIdEpoch epoch maxEpoch cs ns alphaW alphaR ds eps =
    let (cs', ns') = foldl' (structIdIteration alphaW alphaR) (cs, ns) ds
        diff = avgBy (\(oldC, newC) -> sqrtEuclidean oldC newC) $
                toList (Sqnc.zip cs cs')
        epochRatio = (fromIntegral epoch) / (fromIntegral maxEpoch)
        alphaW' = traceShow ns' alphaW * (1.0 - epochRatio)
        alphaR' = alphaR * (1.0 - epochRatio)
    in if diff <= eps || epoch == maxEpoch
       then (cs', epoch, diff)
       else structIdEpoch (epoch + 1) maxEpoch cs' ns' alphaW' alphaR' ds eps

identifyModel :: [ProcessedDataItem] -> IO Model
identifyModel dataset = do
        let startNs = Sqnc.replicate clusterCountLimit 1
        let rand = \_ -> randomRIO (0.0, 1.0)
        startCs <- forM (Sqnc.fromList [1..clusterCountLimit]) $
                   \_ -> forM [1..4] rand
        putStrLn $ show startCs
        let (cs, epochs, diff) = structIdEpoch 1 maxEpoch startCs startNs alphaW alphaR dataset epsilon
        putStrLn $ "Epochs required: " ++ show epochs
        putStrLn $ "Diff achieved: " ++ show diff
        putStrLn $ show cs
        let filteredCs = toList (Sqnc.filter (all insideZeroOne) cs)
        putStrLn $ "Clusters before: " ++ (show $ length cs) ++ " after: " ++
                (show $ length filteredCs)
        if length filteredCs == 0
        then error "No clusters left"
        else do
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
        findAs cs c = let neighborDists = sort $ map (euclidean c) cs
                          sqrDist = if length neighborDists >= 2
                                    then head (tail neighborDists)
                                    else 0.1
                      in replicate 4 (sqrDist / r)
        findB ftrs rspns (a, c) = if denom == 0
                                    then 0.0
                                    else numer / denom
            where step (numer', denom') ((x, z, u, v), y) =
                        let alpha = antecedent ([x, z, u, v], a, c)
                        in (numer' + alpha * y, denom' + alpha)
                  (numer, denom) = foldl' step (0.0, 0.0) $ zip ftrs rspns


costFunction :: Int -> Int -> ProcessedDataset -> [Double] -> Double
costFunction clusterCount featureCount dataset xs =
    let model = fromParamsVector (xs, clusterCount, featureCount)
        localCost (x, z, u, v, y) = (predict model (x, z, u, v) - y) ** 2 
    in avgBy localCost dataset

constraints :: Int -> Int -> [Double] -> [Bool]
constraints clusterCount featureCount xs =
        checkA (clusterCount * featureCount) xs
    where
        checkA 0 ys = checkC (clusterCount * featureCount) ys
        checkA count (y:ys) = (y <= 0) : checkA (count - 1) ys

        checkC 0 ys = checkB clusterCount ys
        checkC count (y:ys) = outsideZeroOne y : checkC (count - 1) ys

        checkB 0 [] = []
        checkB count (y:ys) = notAClass y : checkB (count - 1) ys
            where notAClass y = y < 0.0 || y > 3.0

        outsideZeroOne = not . insideZeroOne


psoIteration :: Int -> Int -> Double ->
    [[Double]] -> [[Double]] -> [[Double]] -> [Double] ->
    ([Double] -> Double) -> ([Double] -> [Bool]) -> IO ([Double], Int, Double)
psoIteration iter maxIter costThreshold xs vs ps pg cost checkConstr = do
        rands1 <- forM [1..swarmSize] $ \_ -> randomRIO (0.0, 1.0)
        rands2 <- forM [1..swarmSize] $ \_ -> randomRIO (0.0, 1.0)
        let newVs = map stepV $ zip5 xs vs ps rands1 rands2
        let (newXs, correctedVs) = unzip $ map stepX $ zip xs newVs
        let newPs = map chooseLessCostly $ zip ps xs
        let newPg = minimumBy (compare `on` cost) (pg : xs)
        let newCost = cost newPg
        if newCost < costThreshold || iter == maxIter
        then return (newPg, iter, newCost)
        else psoIteration (iter + 1) maxIter costThreshold
                    newXs correctedVs newPs newPg cost checkConstr
    where
        w = 0.5
        c1 = 0.5
        c2 = 0.5
        swarmSize = length xs

        stepV (x, v, p, rand1, rand2) = map stepVComponent $ zip4 x v p pg
            where stepVComponent (xe, ve, pe, pge) = w * ve +
                        c1 * rand1 * (pe - xe) + c2 * rand2 * (pge - xe)

        stepX (x, v) = let newX = map (\(xe, ve) -> xe + ve) $ zip x v
                           brokenConstraints = checkConstr newX
                       in (newX, v) if any id brokenConstraints
                          then stepX (x, map (* 0.5) v)
                          else (newX, v)

        chooseLessCostly (a, b) | cost a < cost b = a
                                | otherwise = b

optimizeModel :: Model -> ProcessedDataset -> IO Model
optimizeModel model dataset = do
        let (x0, clusterCount, featureCount) = paramsVector model
        let n = length x0
        let rand = \_ -> randomRIO (0.0, 1.0)
        otherXs <- forM [1..(swarmSize - 1)] $
                   \_ -> forM [1..n] rand
        let xs = x0 : otherXs
        let vs = replicate swarmSize (replicate n 0.0)
        let cost = costFunction clusterCount featureCount dataset
        let pg = minimumBy (compare `on` cost) xs

        (optimizedParams, iters, diff) <- psoIteration 1 maxIterations costThreshold xs vs
            xs pg cost (constraints clusterCount featureCount)
        putStrLn $ "Optimization steps: " ++ (show iters) ++ ", eps: " ++ (show diff)
        return (fromParamsVector (optimizedParams, clusterCount, featureCount))
    where
        swarmSize = 20
        maxIterations = 20
        costThreshold = 0.05

