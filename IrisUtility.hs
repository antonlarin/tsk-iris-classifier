module IrisUtility (
    shuffle,
    splitWith,
    extractFolds,
    avgBy,
    avg,
    insideZeroOne,
    outsideZeroOne,
    euclidean,
    sqrtEuclidean,
    d,

    Parameters,
    parseParams,
    clusterCountLimit,
    alphaW,
    alphaR,
    maxEpochSID,
    epsSID,
    swarmSize,
    omega,
    c1,
    c2,
    maxIterationPSO,
    epsPSO
    ) where

import Control.Monad (forM)
import Data.Array.IO (IOArray, newListArray, readArray, writeArray)
import Data.Foldable (foldl')
import System.Random (randomRIO)

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
          listToArray size ys = newListArray (1, size) ys

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

outsideZeroOne :: Double -> Bool
outsideZeroOne = not . insideZeroOne

euclidean :: [Double] -> [Double] -> Double
euclidean xs ys | length xs /= length ys = error "Euclidean: uneven lists"
                | otherwise = foldl' step 0.0 $ zip xs ys
    where step acc (x, y) = acc + (x - y) ** 2

sqrtEuclidean :: [Double] -> [Double] -> Double
sqrtEuclidean xs ys = sqrt $ euclidean xs ys

d :: [Double] -> [Double] -> Int -> Int -> Double
d x c n nSum = (fromIntegral n) * euclidean c x / (fromIntegral nSum)


data Parameters = Parameters {
    clusterCountLimit :: Int,
    alphaW :: Double,
    alphaR :: Double,
    maxEpochSID :: Int,
    epsSID :: Double,
    swarmSize :: Int,
    omega :: Double,
    c1 :: Double,
    c2 :: Double,
    maxIterationPSO :: Int,
    epsPSO :: Double
}

parseParams :: [String] -> Parameters
parseParams strParams | length strParams /= 11 = error "11 args required"
                      | otherwise =
    let clusterCountLimit = read (strParams !! 0) :: Int
        alphaW = read (strParams !! 1) :: Double
        alphaR = read (strParams !! 2) :: Double
        maxEpochSID = read (strParams !! 3) :: Int
        epsSID = read (strParams !! 4) :: Double
        swarmSize = read (strParams !! 5) :: Int
        omega = read (strParams !! 6) :: Double
        c1 = read (strParams !! 7) :: Double
        c2 = read (strParams !! 8) :: Double
        maxIterationPSO = read (strParams !! 9) :: Int
        epsPSO = read (strParams !! 10) :: Double
    in Parameters clusterCountLimit alphaW alphaR maxEpochSID epsSID
            swarmSize omega c1 c2 maxIterationPSO epsPSO

