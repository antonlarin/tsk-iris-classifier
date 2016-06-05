module IrisDefs (
    Iris,
    irisFromString,
    irisToDouble,
    irisFromDouble,

    DataItem,
    Dataset,
    ProcessedDataItem,
    ProcessedDataset,
    procItemResponse,
    procItemFeatures
    ) where

-- iris
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

irisToDouble :: Iris -> Double
irisToDouble Setosa = 2.5
irisToDouble Versicolor = 0.5
irisToDouble Virginica = 1.5

irisFromDouble :: Double -> Iris
irisFromDouble val | 2 <= val && val < 3 = Setosa
                   | 0 <= val && val < 1 = Versicolor
                   | 1 <= val && val <= 2 = Virginica
                   | otherwise = error "Can't convert Double to Iris "


-- data item/set
type DataItem = (Double, Double, Double, Double, Iris)
type Dataset = [DataItem]

type ProcessedDataItem = (Double, Double, Double, Double, Double)
type ProcessedDataset = [ProcessedDataItem]

procItemResponse :: ProcessedDataItem -> Double
procItemResponse (_, _, _, _, resp) = resp

procItemFeatures :: ProcessedDataItem -> (Double, Double, Double, Double)
procItemFeatures (a, b, c, d, _) = (a, b, c, d)

