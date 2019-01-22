module Volr.Evaluator where

import Control.Applicative
import Control.Monad.Except
import Control.Monad.State.Lazy

import Data.Either
import qualified Data.Map.Strict as Map

import Volr.AST

type Error = String

data TermState = TermState { types :: Context, store :: Store }
  deriving (Eq, Show)
type EvalState = ExceptT Error (State TermState)

emptyState = TermState Map.empty Map.empty

eval :: Term -> Either Error Term
eval term = evalState (runExceptT (evalTyped term)) emptyState
  where
    evalTyped term = do
      untyped <- eval' term
      typeOf untyped *> return untyped

eval' :: Term -> EvalState Term
eval' term =
  case term of
    TmNet n m -> return $ TmNet n m
    TmSeq t1 t2 -> do
        t1' <- eval' t1
        t2' <- eval' t2
        return $ TmSeq t1' t2'
    TmPar t1 t2 -> do
        t1' <- eval' t1
        t2' <- eval' t2
        return $ TmPar t1' t2'
    TmRef n -> do
        state <- get
        case store state Map.!? n of
          Nothing -> throwError $ "Could not find reference of name " ++ n
          Just m -> return m 
    TmLet name t1 t2 -> do
        state <- get
        t1' <- eval' t1
        put $ state { store = Map.insert name t1' (store state) }
        t2' <- eval' t2
        put $ state
        return t2'

-- | Tests whether a given term is a value
isVal :: Term -> Bool
isVal (TmNet _ _) = True
isVal _ = False

typeOf :: Term -> EvalState Type
typeOf term = 
  case term of
    TmNet n m -> return $ TyNetwork n m
    TmSeq t1 t2 -> do
      leftOut <- sizeRight t1
      rightIn <- sizeLeft t2
      if leftOut == rightIn then do
        leftIn <- sizeLeft t1
        rightOut <- sizeRight t2
        return $ TyNetwork leftIn rightOut 
      else
        throwError $ "Type error: Incompatible network sizes. Output " ++ 
                     (show leftOut) ++ " should be equal to input " ++ (show rightIn)
    TmPar t1 t2 -> do
      left1 <- sizeLeft t1 
      left2 <- sizeLeft t2
      if left1 == left2 then do
        right1 <- sizeRight t1
        right2 <- sizeRight t2
        return $ TyNetwork left1 (right1 + right2)
      else
        throwError $ "Type error: Parallel networks must share input sizes, got " ++ 
                     (show left1) ++ " and " ++ (show left2)
    TmLet name t1 t2 -> do
      state <- get
      t1' <- eval' t1
      let innerState = state { store = Map.insert name t1' (store state) }
      evalState (return $ typeOf t2) innerState

sizeLeft :: Term -> EvalState Int
sizeLeft term = 
  case term of 
    TmNet m _ -> return m 
    TmSeq t1 t2 -> sizeLeft t1 
    TmPar t1 t2 -> sizeLeft t1
    TmRef n -> do
      state <- get
      case store state Map.!? n of
        Nothing -> throwError $ "Unknown reference " ++ n
        Just e -> sizeLeft e
    _ -> throwError $ "Cannot extract size from term " ++ (show term)

sizeRight :: Term -> EvalState Int
sizeRight term = 
  case term of 
    TmNet _ m -> return m 
    TmSeq t1 t2 -> sizeRight t2
    TmPar t1 t2 -> (+) <$> sizeRight t1 <*> sizeRight t2
    TmRef n -> do
      state <- get
      case store state Map.!? n of
        Nothing -> throwError $ "Unknown reference " ++ n
        Just e -> sizeRight e
    _ -> throwError $ "Cannot extract size from term " ++ (show term)

    
