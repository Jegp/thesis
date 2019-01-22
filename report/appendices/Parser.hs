module Volr.Parser where

import Control.Applicative hiding (many, some)
import Control.Monad.State.Lazy

import Data.Bifunctor  
import Data.Functor
import qualified Data.Map as Map
import Data.Maybe (isJust)

import Text.Megaparsec
import qualified Text.Megaparsec.Char as Char
import qualified Text.Megaparsec.Char.Lexer as Lexer
import qualified Text.Megaparsec.Pos as Pos

import Volr.AST
import Volr.Evaluator

type SyntaxError = ParseError (Token String) String
type Parser = Parsec String String

parse :: String -> Either String Term
parse code = 
  first show (runParser parseTerm "" code) >>= eval

parseTerm :: Parser Term
parseTerm = (lexeme $ choice
  [ TmNet <$> (symbol "Net" *> integer) <*> integer
  , TmPar <$> (symbol "Par" *> (parens parseTerm)) <*> (parens parseTerm)
  , TmSeq <$> (symbol "Seq" *> (parens parseTerm)) <*> (parens parseTerm)
  , TmRef <$> (symbol "Ref" *> (name))
  , TmLet <$> (symbol "Let" *> (name)) <*> (symbol "=" *> parseTerm)
                           <*> (symbol "in" *> parseTerm)
  ]) <* (optional eof)

integer :: Parser Int
integer = lexeme Lexer.decimal

parens :: Parser a -> Parser a
parens = between (symbol "(") (symbol ")")

name :: Parser String
name = lexeme (many Char.alphaNumChar)

symbol :: String -> Parser String
symbol = Lexer.symbol sc

lexeme :: Parser a -> Parser a
lexeme = Lexer.lexeme sc

sc :: Parser ()
sc = Lexer.space Char.space1 lineCmnt blockCmnt
  where
    lineCmnt  = Lexer.skipLineComment "//"
    blockCmnt = Lexer.skipBlockComment "/*" "*/"
