import Data.List (zip5, unzip5)
import System.IO
import System.Random (mkStdGen, setStdGen)
import System.Environment (getArgs)

import IrisUtility
import IrisDefs
import IrisModel
import IrisStructuralIdentification
import IrisOptimization

main :: IO ()
main = do
    input <- openFile "iris.data" ReadMode
    dataset <- readDataset input
    hClose input
    stringParams <- getArgs
    let params = parseParams stringParams
    let seed = 123
    setStdGen $ mkStdGen seed
    shuffledDataset <- shuffle dataset
    let processedDataset = preprocess shuffledDataset
    let folds = 10
    let trainTestPairs = crossValidationSplit folds processedDataset
    scores <- mapM (buildAndTestModel params) trainTestPairs
    let (preOptScores, postOptScores) = unzip scores
    let meanPreOptScore = avg preOptScores
    let meanPostOptScore = avg postOptScores
    putStrLn $ "Pre opt: " ++ show meanPreOptScore
    putStrLn $ "Post opt: " ++ show meanPostOptScore


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

buildAndTestModel :: Parameters -> (ProcessedDataset, ProcessedDataset) ->
        IO (Double, Double)
buildAndTestModel params (train, test) =
    identifyModel params train >>= \model ->
    optimizeModel params model train >>= \model' ->
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

