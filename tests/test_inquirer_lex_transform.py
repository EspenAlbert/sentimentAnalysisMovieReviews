from unittest import TestCase

from samr.inquirer_lex_transform import InquirerLexTransform


class TestInquirerLexTransform(TestCase):
    def test_empty(self):
        m = InquirerLexTransform()
        Z = m.transform([])
        self.assertEqual(len(Z), 0)

    def test_fit_returns_self(self):
        m = InquirerLexTransform()
        s = m.fit([])
        self.assertEqual(s, m)

    def test_simple(self):
        # X = ["This was a good summer", "The food was awful", "what is good good good for the goose"]
        X = ["This was a good summer", "The food was awful"]
        m = InquirerLexTransform()
        Z = m.transform(X)
        print(Z)
        self.assertEqual(len(Z), 2)
        self.assertTrue(isinstance(Z[0], str) and isinstance(Z[1], str))
        self.assertIn("positiv", Z[0].lower())
        self.assertIn("negativ", Z[1].lower())
        self.assertNotIn("good", Z[0].lower())
        self.assertNotIn("awful", Z[1].lower())
