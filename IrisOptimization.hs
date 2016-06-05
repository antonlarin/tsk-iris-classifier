module IrisOptimization (
    optimizeModel
    ) where

import Control.Monad (forM)
import Data.Foldable (minimumBy)
import Data.Function (on)
import Data.List (zip4, zip5)
import System.Random (randomRIO)

import IrisDefs
import IrisModel
import IrisUtility

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

psoIteration :: Parameters -> Int -> Int -> Double ->
    [[Double]] -> [[Double]] -> [[Double]] -> [Double] ->
    ([Double] -> Double) -> ([Double] -> [Bool]) -> IO ([Double], Int, Double)
psoIteration params iter maxIter costThreshold xs vs ps pg cost checkConstr =
    do
        rands1 <- forM [1..swarmSize params] $ \_ -> randomRIO (0.0, 1.0)
        rands2 <- forM [1..swarmSize params] $ \_ -> randomRIO (0.0, 1.0)
        let newVs = map stepV $ zip5 xs vs ps rands1 rands2
        let (newXs, correctedVs) = unzip $ map stepX $ zip xs newVs
        let newPs = map chooseLessCostly $ zip ps xs
        let newPg = minimumBy (compare `on` cost) (pg : xs)
        let bestCost = cost newPg
        if bestCost < costThreshold || iter == maxIter
        then return (newPg, iter, bestCost)
        else psoIteration params (iter + 1) maxIter costThreshold
                    newXs correctedVs newPs newPg cost checkConstr
    where
        stepV (x, v, p, rand1, rand2) = map stepVComponent $ zip4 x v p pg
            where stepVComponent (xe, ve, pe, pge) = (omega params) * ve +
                        (c1 params) * rand1 * (pe - xe) +
                        (c2 params) * rand2 * (pge - xe)

        stepX (x, v) = let newX = map (\(xe, ve) -> xe + ve) $ zip x v
                           brokenConstraints = checkConstr newX
                       in if any id brokenConstraints
                          then stepX (x, map (* 0.5) v)
                          else (newX, v)

        chooseLessCostly (a, b) | cost a < cost b = a
                                | otherwise = b

optimizeModel :: Parameters -> Model -> ProcessedDataset -> IO Model
optimizeModel params model dataset = do
    let (x0, clusterCount, featureCount) = paramsVector model
    let n = length x0
    let rand = \_ -> randomRIO (0.0, 1.0)
    otherXs <- forM [1..swarmSize params - 1] $
               \_ -> forM [1..n] rand
    let xs = x0 : otherXs
    let vs = replicate (swarmSize params) (replicate n 0.0)
    let cost = costFunction clusterCount featureCount dataset
    let pg = minimumBy (compare `on` cost) xs

    (optimizedParams, iters, bestCost) <- psoIteration params 1
        (maxIterationPSO params) (epsPSO params) xs vs xs pg cost
        (constraints clusterCount featureCount)
    putStrLn $ ("Optimization steps: " ++ (show iters) ++
                ", best cost: " ++ (show bestCost))
    return (fromParamsVector (optimizedParams, clusterCount, featureCount))

