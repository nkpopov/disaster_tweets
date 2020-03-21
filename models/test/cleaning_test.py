#!/usr/bin/env python3

import sys
sys.path.append("/home/popovni/_kaggle/_disaster_tweets/baseline")

import pandas as pd
import unittest
import cleaning


class PandasDataFrameTest(unittest.TestCase):
    def setUp(self):
        def assertDataFrameEqual(a, b, msg):
            pd.testing.assert_frame_equal(a, b)
        self.addTypeEqualityFunc(pd.DataFrame, assertDataFrameEqual)

        
class RemoveUrlsTest(PandasDataFrameTest):
    def test_dataframe_without_urls_returns_identical(self):
        target = pd.DataFrame({"text": ["a b c", "d ef"]})
        result = cleaning.remove_urls(target)
        self.assertEqual(result, target)

    def test_dataframe_with_urls_returns_urls_removed(self):
        input_ = pd.DataFrame({"text": ["a https://google.com", "aa aa"]})
        target = pd.DataFrame({"text": ["a ", "aa aa"]})
        result = cleaning.remove_urls(input_)
        self.assertEqual(result, target)


class RemoveHtmlTagsTest(PandasDataFrameTest):
    def test_dataframe_without_html_tags_returns_identical(self):
        target = pd.DataFrame({"text": ["a b c", "d ef"]})
        result = cleaning.remove_html_tags(target)
        self.assertEqual(result, target)

    def test_dataframe_with_html_tags_returns_tags_removed(self):
        input_ = pd.DataFrame({"text": ["a b <c>", "d ef"]})
        target = pd.DataFrame({"text": ["a b ", "d ef"]})
        result = cleaning.remove_html_tags(input_)
        self.assertEqual(result, target)


class RemoveEmojisTest(PandasDataFrameTest):
    def test_dataframe_without_emojis_returns_identical(self):
        target = pd.DataFrame({"text": ["a b c", "d ef"]})
        result = cleaning.remove_emojis(target)
        self.assertEqual(result, target)

    def test_dataframe_with_emojis_returns_emojis_removed(self):
        input_ = pd.DataFrame({"text": ["a b c " + u"\U0001F600", "d ef"]})
        target = pd.DataFrame({"text": ["a b c ", "d ef"]})
        result = cleaning.remove_emojis(input_)
        self.assertEqual(result, target)


class RemovePunctuationTest(PandasDataFrameTest):
    def test_dataframe_without_punctuation_returns_identical(self):
        target = pd.DataFrame({"text": ["a b c", "d ef"]})
        result = cleaning.remove_punctuation(target)
        self.assertEqual(result, target)

    def test_dataframe_with_punctuation_returns_punctuation_removed(self):
        input_ = pd.DataFrame({"text": ["a b c,.", "d ef;"]})
        target = pd.DataFrame({"text": ["a b c", "d ef"]})
        result = cleaning.remove_punctuation(input_)
        self.assertEqual(result, target)
        

class CorrectSpellingTest(PandasDataFrameTest):
    def test_dataframe_without_spelling_mistakes_returns_identical(self):
        target = pd.DataFrame({"text": ["cat", "identical"]})
        result = cleaning.correct_spelling(target)
        self.assertEqual(result, target)
        
    def test_dataframe_with_spelling_mistakes_returns_corrected_text(self):
        input_ = pd.DataFrame({"text": ["cat", "idintical"]})
        target = pd.DataFrame({"text": ["cat", "identical"]})
        result = cleaning.correct_spelling(target)
        self.assertEqual(result, target)


class ReplaceWordsNotCoveredByEmbeddingsTest(PandasDataFrameTest):
    def test_dataframe_without_not_covered_words_returns_identical(self):
        target = pd.DataFrame({"text": ["cat", "identical"]})
        result = cleaning.replace_words_not_covered_by_embeddings(target)
        self.assertEqual(result, target)

    def test_dataframe_with_not_covered_words_returns_correct_text(self):
        input_ = pd.DataFrame({"text": ["cat", "test typhoondevastated", "gbbo dista"]})
        target = pd.DataFrame({"text": ["cat", "test typhoon devastated", "great british bake off parties"]})
        result = cleaning.replace_words_not_covered_by_embeddings(target)
        self.assertEqual(result, target)


if __name__ == "__main__":
    unittest.main()

