module IrisStructuralIdentification (
    identifyModel
    ) where

import Control.Monad (forM)
import Data.Foldable (foldl', toList)
import Data.Function (on)
import Data.List (sort)
import qualified Data.Sequence as Sqnc
import System.Random (randomRIO)

import IrisDefs
import IrisModel
import IrisUtility

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
        alphaW' = alphaW * (1.0 - epochRatio)
        alphaR' = alphaR * (1.0 - epochRatio)
    in if diff <= eps || epoch == maxEpoch
       then (cs', epoch, diff)
       else structIdEpoch (epoch + 1) maxEpoch cs' ns' alphaW' alphaR' ds eps

identifyModel :: Parameters -> [ProcessedDataItem] -> IO Model
identifyModel params dataset = do
        let startNs = Sqnc.replicate (clusterCountLimit params) 1
        let rand = \_ -> randomRIO (0.0, 1.0)
        startCs <- forM (Sqnc.fromList [1..clusterCountLimit params]) $
                   \_ -> forM [1..4] rand
        let (cs, epochs, diff) = (structIdEpoch 1 (maxEpochSID params)
                startCs startNs (alphaW params) (alphaR params)
                dataset (epsSID params))
        putStrLn $ ("SID: Epochs: " ++ (show epochs) ++
                    ", diff: " ++ (show diff))
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
            return (TSK0 (length as) as filteredCs bs)
    where
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

