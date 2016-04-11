#  Explain and drop some links here
from collections import namedtuple, defaultdict
import csv
import os

from samr.transformations import StatelessTransform
from samr.settings import DATA_PATH


FIELDS = ("Entry, Source, Positiv, Negativ, Pstv, Affil, Ngtv, Hostile, Strong,"
          " Power, Weak, Submit, Active, Passive, Pleasur, Pain, Feel, Arousal,"
          " EMOT, Virtue, Vice, Ovrst, Undrst, Academ, Doctrin, Econ, Exch, "
          "ECON, Exprsv, Legal, Milit, Polit, POLIT, Relig, Role, COLL, Work, "
          "Ritual, SocRel, Race, Kin, MALE, Female, Nonadlt, HU, ANI, PLACE, "
          "Social, Region, Route, Aquatic, Land, Sky, Object, Tool, Food, "
          "Vehicle, BldgPt, ComnObj, NatObj, BodyPt, ComForm, COM, Say, Need, "
          "Goal, Try, Means, Persist, Complet, Fail, NatrPro, Begin, Vary, "
          "Increas, Decreas, Finish, Stay, Rise, Exert, Fetch, Travel, Fall, "
          "Think, Know, Causal, Ought, Perceiv, Compare, Eval, EVAL, Solve, "
          "Abs, ABS, Quality, Quan, NUMB, ORD, CARD, FREQ, DIST, Time, TIME, "
          "Space, POS, DIM, Rel, COLOR, Self, Our, You, Name, Yes, No, Negate, "
          "Intrj, IAV, DAV, SV, IPadj, IndAdj, PowGain, PowLoss, PowEnds, "
          "PowAren, PowCon, PowCoop, PowAuPt, PowPt, PowDoct, PowAuth, PowOth, "
          "PowTot, RcEthic, RcRelig, RcGain, RcLoss, RcEnds, RcTot, RspGain, "
          "RspLoss, RspOth, RspTot, AffGain, AffLoss, AffPt, AffOth, AffTot, "
          "WltPt, WltTran, WltOth, WltTot, WlbGain, WlbLoss, WlbPhys, WlbPsyc, "
          "WlbPt, WlbTot, EnlGain, EnlLoss, EnlEnds, EnlPt, EnlOth, EnlTot, "
          "SklAsth, SklPt, SklOth, SklTot, TrnGain, TrnLoss, TranLw, MeansLw, "
          "EndsLw, ArenaLw, PtLw, Nation, Anomie, NegAff, PosAff, SureLw, If, "
          "NotLw, TimeSpc, FormLw, Othtags, Defined")

InquirerLexEntry = namedtuple("InquirerLexEntry", FIELDS)
FIELDS = InquirerLexEntry._fields


class InquirerLexTransform(StatelessTransform):
    _corpus = []
    _use_fields = [FIELDS.index(x) for x in "Positiv Negativ IAV Strong Pstv Ngtv Weak Active Passive".split()]

    def transform(self, X, y=None):
        """
        `X` is expected to be a list of `str` instances containing the phrases.
        Return value is a list of `str` containing different amounts of the
        words "Positiv_Positiv", "Negativ_Negativ", "IAV_IAV", "Strong_Strong"
        based on the sentiments given to the input words by the Hardvard
        Inquirer lexicon.
        """
        corpus = self._get_corpus()
        result = []
        for phrase in X:
            newphrase = []
            for word in phrase.split():
                newphrase.extend(corpus.get(word.lower(), []))
            result.append(" ".join(newphrase))
        return result

    def _get_corpus(self):
        """
        Private method used to cache a dictionary with the Harvard Inquirer
        corpus.
        ** Stores in the following manner:
        corpus = {'abandonment': ['Negativ_Negativ'], abandon: [Negativ_Negativ, "IAV_IAV"], .....
        """
        if not self._corpus:
            corpus = defaultdict(list)
            it = csv.reader(open(os.path.join(DATA_PATH, "inquirerbasicttabsclean")),
                            delimiter="\t")
            next(it)  # Drop header row
            for row in it:
                entry = InquirerLexEntry(*row)
                xs = []
                for i in self._use_fields:
                    name, x = FIELDS[i], entry[i]
                    if x:
                        xs.append("{}_{}".format(name, x))
                name = entry.Entry.lower()
                if "#" in name: #If a entry have multiple meanings, eg: absent#1, absent#2. Defs:  27% adj: Perfect, without limitation or qualification, 73% adv: "Absolutely"--without qualification, certainly, totally
                    #Example 2: BALL#1, BALL#2, BALL#3: | 50% noun-adj: A spherical object, a kind of bullet (0), | 25% noun-adj: A game which is played with a ball | 4% noun: A formal social dance
                    name = name[:name.index("#")]
                corpus[name].extend(xs)
            self._corpus.append(dict(corpus))
        return self._corpus[0]

