import System.IO
import Data.Char (toLower)
import Data.List (unzip5, zip5)

main :: IO ()
main = do
    input <- openFile "iris.data" ReadMode
    dataset <- readDataset input []
    let preprocessedDataset = preprocess dataset
    hClose input


type DataItem = (Double, Double, Double, Double, Iris)

readDataset :: Handle -> [DataItem] -> IO [DataItem]
readDataset input xs = do
    endOfInput <- hIsEOF input
    if endOfInput
    then return xs
    else do
        line <- hGetLine input
        let entries = splitWith ',' line
        let [a, b, c, d] = map (\n -> read n :: Double) $ take 4 entries
        let entryClass = head $ drop 4 entries
        let dataItem = (a, b, c, d, irisFromString entryClass)
        readDataset input (dataItem : xs)


splitWith :: Char -> String -> [String]
splitWith _ [] = []
splitWith sep lines = front : splitWith sep back
            where notSep = (/= sep)
                  front = takeWhile notSep lines
                  backWithSep = dropWhile notSep lines
                  back = if null backWithSep
                         then []
                         else tail backWithSep

data Iris = Setosa
          | Versicolor
          | Virginica
          deriving Show

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


type ProcessedDataItem = (Double, Double, Double, Double, Double)

preprocess :: [DataItem] -> [ProcessedDataItem]
preprocess dataset = zip5 as' bs' cs' ds' irises'
            where (as, bs, cs, ds, irises) = unzip5 dataset
                  irises' = map irisToDouble irises
                  [as', bs', cs', ds'] = map normalize [as, bs, cs, ds]
                  normalize xs = let min = minimum xs
                                     max = maximum xs
                                     invSpan = 1.0 / (max - min)
                                 in map (\x -> invSpan * (x - min)) xs 

