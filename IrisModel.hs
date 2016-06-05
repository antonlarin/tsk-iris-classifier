module IrisModel (
    Model (TSK0),
    paramsVector,
    fromParamsVector,
    antecedent,
    predict
    ) where

-- TSK0 model
data Model = TSK0 Int [[Double]] [[Double]] [Double]
             deriving Show

paramsVector :: Model -> ([Double], Int, Int)
paramsVector (TSK0 clusterCount as cs bs) =
        (flatten as ++ flatten cs ++ bs, clusterCount, length (head as))
    where flatten xss = foldr (++) [] xss

fromParamsVector :: ([Double], Int, Int) -> Model
fromParamsVector (xs, clusterCount, featureCount) =
    let
        (as, xs') = unflatten clusterCount xs
        (cs, bs) = unflatten clusterCount xs'
    in TSK0 clusterCount as cs bs
    where
        unflatten 0 xs = ([], xs)
        unflatten clustersLeft xs = (cluster : otherACs, otherVec)
            where (cluster, rest) = splitAt featureCount xs
                  (otherACs, otherVec) = unflatten (clustersLeft - 1) rest

antecedent :: ([Double], [Double], [Double]) -> Double
antecedent (xs, as, cs) = product $ map antecedentConjunct $ zip3 xs as cs
    where antecedentConjunct (x, a, c) = exp ((x - c) ** 2 / (-2 * a * a))

predict :: Model -> (Double, Double, Double, Double) -> Double
predict (TSK0 rulesCount aSets cSets bs) (x, y, z, w) =
    let xSets = replicate rulesCount [x, y, z, w]
        antecedents = map antecedent $ zip3 xSets aSets cSets
        numerator = sum $ map tupleProduct $ zip antecedents bs
        denominator = sum antecedents
    in if denominator == 0
       then 0.0
       else numerator / denominator
    where tupleProduct (p, b) = p * b

