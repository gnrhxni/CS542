#/usr/bin/python

import os
import re
import sys
from collections import namedtuple
import numpy as np
import csv

Dictionary_Element = namedtuple("Dictionary_Element", 
                                "word phonemes stress flag")

here = os.path.dirname(os.path.abspath(__file__))
defaultdatafile = os.path.join(here, "nettalk.data")

def testit():
    for entry in dictionary():
        print(entry.word); 
        outarray = outputUnits(entry);
        print(outarray);
        sys.exit(1);


def dictionary(datafile=defaultdatafile):
    with open(datafile) as data_fp:
        for line in data_fp:
            match = re.match(r'([a-z]+)\t(\S+)\t([\d<>]+)\t(\d+)', line)
            if match:
                yield Dictionary_Element._make(match.groups())


def wordstream(windowsize=7, input_entries=None, padchar='-'):
    """Note: middle of each window is (windowsize/2)+1, since python
    automatically floors uneven integer division.
    """
    even = bool(windowsize % 2 == 0)
    if even:
        lmargin, rmargin = windowsize/2, windowsize/2
    if not even:
        lmargin, rmargin = windowsize/2, (windowsize/2)+1

    if not input_entries:
        input_entries = dictionary()

    for entry in input_entries:
        word = entry.word
        ret = list()
        for i in range(len(word)):
            chunk = word[max(i-lmargin,0):i+rmargin]
            lpad = -min(i-lmargin, 0)
            rpad = windowsize - len(chunk) - lpad
            ret.append( 
                padchar*(lpad) + chunk + padchar*(rpad)
            )
                
        yield ret



def createFeatureTable(filename):
    features = dict();
    featureToUnit = dict();
    csvfile =  open(filename, 'r');
    reader = csv.reader(csvfile, delimiter=',', quotechar='"');
    for row in reader: 
        if (1 != len(row[0])): continue;
        for i in range(3,len(row)):
            if (row[i] not in featureToUnit and len(row[i])):
                featureToUnit[row[i]] = len(featureToUnit);
    print(featureToUnit);
    csvfile.seek(0);
    reader = csv.reader(csvfile, delimiter=',', quotechar='"');
    for row in reader: 
        grapheme = row[0];
        if (1 != len(grapheme)): continue;
        if (grapheme not in features):
          features[grapheme] = [];
        for i in range(3,len(row)):
            if (len(row[i])):
                intfeature = featureToUnit[row[i]];
                features[grapheme].append(intfeature);
    return (features, len(featureToUnit));   



articFeatureNames = {'Pause': 27, 'Full Stop': 28, 'Silent': 25, 'Glide': 13, 'Palatal': 20, 'Nasal': 18, 'Unvoiced': 10, 'Low': 0, 'Alveolar': 8, 'Medium': 6, 'Dental': 16, 'Stop': 5, 'Central1': 21, 'Tensed': 1, 'Central2': 2, 'Front1': 15, 'Front2': 9, 'Central': 24, 'Liquid': 17, 'Velar': 7, 'Back2': 19, 'Elide': 26, 'Back1': 23, 'Fricative': 11, 'Labial': 4, 'High': 14, 'Affricative': 22, 'Glottal': 12, 'Voiced': 3}

articFeatures = {'!': [10, 4, 16, 22], '#': [3, 20, 7, 22], '*': [3, 13, 15, 0, 21, 24], '-': [25, 26], '.': [27, 28], 'A': [6, 1, 9, 21], '@': [0, 9], 'C': [10, 20, 22], 'E': [6, 15, 9], 'D': [3, 16, 11], 'G': [3, 7, 18], 'I': [14, 15], 'K': [10, 20, 11, 7, 22], 'J': [3, 7, 18], 'M': [3, 16, 18], 'L': [3, 8, 17], 'O': [6, 1, 21, 2], 'N': [3, 20, 18], 'Q': [3, 4, 7, 22, 5], 'S': [10, 20, 11], 'R': [3, 7, 17], 'U': [14, 23], 'T': [10, 16, 11], 'Y': [14, 1, 15, 9, 21], 'X': [10, 22, 9, 21], 'Z': [3, 20, 11], '_': [27, 26], '^': [0, 24], 'a': [0, 1, 2], 'c': [6, 7], 'b': [3, 4, 5], 'e': [6, 1, 9], 'd': [3, 8, 5], 'g': [3, 7, 5], 'f': [10, 4, 11], 'i': [14, 1, 15], 'h': [10, 12, 13], 'k': [10, 7, 5], 'm': [3, 4, 18], 'l': [3, 16, 17], 'o': [6, 1, 19], 'n': [3, 8, 18], 's': [10, 8, 11], 'r': [3, 20, 17], 'u': [14, 1, 19], 't': [10, 8, 5], 'w': [3, 4, 13, 14, 6, 1, 2, 23], 'v': [3, 4, 11], 'y': [3, 20, 13], 'x': [6, 2], 'z': [3, 8, 11], '|': [14, 15, 9]}

stressFeatureNames = {'boundary': 4, 'right': 0, 'weak': 3, 'strong': 2, 'left': 1}

stressFeatures = {'1': [2, 3], '0': [3], '2': [2], '<': [0], '_': [0, 1, 4], '>': [1]}

MINSTRESS = len(articFeatureNames);
NUMOUTPUTS = MINSTRESS + len(stressFeatureNames) + 2;
FOREIGN = NUMOUTPUTS - 1;
WEIRD = NUMOUTPUTS - 2;

def outputUnits(entry):
    ret = np.zeros((len(entry.word), NUMOUTPUTS), np.int8)
    for i in range(len(entry.word)):
        phoneme = entry.phonemes[i];
        features = articFeatures[phoneme];
        for f in features: 
            ret[i][f] = 1;
        features = stressFeatures[entry.stress[i]];
        for f in features: 
            ret[i][MINSTRESS + f] = 1;
        if (1 == entry.flag): 
            ret[i][WEIRD] = 1;
        elif (2 == entry.flag): 
            ret[i][FOREIGN] = 1;
        print(phoneme, " ", ret[i]);
    return ret;
    
topK = [
'THE','OF','AND','TO','IN',
'THAT','IS','WAS','HE','FOR',
'IT','WITH','AS','HIS','ON',
'BE','AT','BY','I','THIS',
'HAD','NOT','ARE','BUT','FROM',
'OR','HAVE','AN','THEY','WHICH',
'ONE','YOU','WERE','HER','ALL',
'SHE','THERE','WOULD','THEIR','WE',
'HIM','BEEN','HAS','WHEN','WHO',
'WILL','MORE','NO','IF','OUT',
'SO','SAID','WHAT','UP','ITS',
'ABOUT','INTO','THAN','THEM','CAN',
'ONLY','OTHER','NEW','SOME','TIME',
'COULD','THESE','TWO','MAY','THEN',
'DO','FIRST','ANY','MY','NOW',
'SUCH','LIKE','OUR','OVER','MAN',
'ME','EVEN','MOST','MADE','AFTER',
'ALSO','DID','MANY','BEFORE','MUST',
'THROUGH','BACK','WHERE','MUCH','YOUR',
'WAY','WELL','DOWN','SHOULD','BECAUSE',
'EACH','JUST','THOSE','PEOPLE','HOW',
'TOO','LITTLE','STATE','GOOD','VERY',
'MAKE','WORLD','STILL','OWN','SEE',
'MEN','WORK','LONG','HERE','GET',
'BOTH','BETWEEN','LIFE','BEING','UNDER',
'NEVER','SAME','DAY','ANOTHER','KNOW',
'WHILE','LAST','MIGHT','US','GREAT',
'OLD','YEAR','OFF','COME','SINCE',
'GO','AGAINST','CAME','RIGHT','TAKE',
'THREE','HIMSELF','FEW','HOUSE','USE',
'DURING','WITHOUT','AGAIN','PLACE','AMERICAN',
'AROUND','HOWEVER','HOME','SMALL','FOUND',
'THOUGHT','WENT','SAY','PART','ONCE',
'HIGH','GENERAL','UPON','SCHOOL','EVERY',
'DOES','GOT','UNITED','LEFT','NUMBER',
'COURSE','WAR','UNTIL','ALWAYS','SOMETHING',
'FACT','THOUGH','WATER','LESS','PUBLIC',
'PUT','THINK','ALMOST','HAND','ENOUGH',
'FAR','TOOK','HEAD','YET','GOVERNMENT',
'SYSTEM','SET','BETTER','TOLD','NOTHING',
'NIGHT','END','WHY','FIND','LOOK',
'GOING','POINT','KNEW','NEXT','CITY',
'BUSINESS','GIVE','GROUP','YOUNG','LET',
'ROOM','PRESIDENT','SIDE','SOCIAL','SEVERAL',
'GIVEN','PRESENT','ORDER','NATIONAL','RATHER',
'POSSIBLE','SECOND','FACE','PER','AMONG',
'FORM','OFTEN','EARLY','WHITE','CASE',
'LARGE','BECOME','NEED','BIG','FOUR',
'WITHIN','FELT','ALONG','CHILDREN','SAW',
'BEST','CHURCH','EVER','LEAST','POWER',
'THING','LIGHT','FAMILY','INTEREST','WANT',
'MIND','COUNTRY','AREA','DONE','OPEN',
'GOD','SERVICE','CERTAIN','KIND','PROBLEM',
'THUS','DOOR','HELP','SENSE','WHOLE',
'MATTER','PERHAPS','ITSELF','TIMES','HUMAN',
'LAW','LINE','ABOVE','NAME','EXAMPLE',
'ACTION','COMPANY','LOCAL','SHOW','WHETHER',
'FIVE','HISTORY','GAVE','EITHER','TODAY',
'FEET','ACT','ACROSS','TAKEN','PAST',
'QUITE','HAVING','SEEN','DEATH','BODY',
'EXPERIENCE','REALLY','HALF','WEEK','WORD',
'FIELD','CAR','ALREADY','THEMSELVES','INFORMATION',
'TELL','TOGETHER','SHALL','COLLEGE','PERIOD',
'MONEY','SURE','HELD','KEEP','PROBABLY',
'REAL','FREE','CANNOT','MISS','POLITICAL',
'QUESTION','AIR','OFFICE','BROUGHT','WHOSE',
'SPECIAL','HEARD','MAJOR','AGO','MOMENT',
'STUDY','FEDERAL','KNOWN','AVAILABLE','STREET',
'RESULT','ECONOMIC','BOY','REASON','POSITION',
'CHANGE','SOUTH','BOARD','INDIVIDUAL','JOB',
'SOCIETY','WEST','CLOSE','TURN','LOVE',
'TRUE','COMMUNITY','FULL','FORCE','COURT',
'SEEM','COST','AM','WIFE','FUTURE',
'AGE','VOICE','CENTER','WOMAN','COMMON',
'CONTROL','NECESSARY','POLICY','FRONT','SIX',
'GIRL','CLEAR','FURTHER','LAND','ABLE',
'FEEL','PARTY','MUSIC','PROVIDE','MOTHER',
'UNIVERSITY','EDUCATION','EFFECT','LEVEL','CHILD',
'SHORT','RUN','STOOD','TOWN','MILITARY',
'MORNING','TOTAL','OUTSIDE','FIGURE','RATE',
'ART','CENTURY','CLASS','NORTH','LEAVE',
'THEREFORE','PLAN','TOP','SOUND','EVIDENCE',
'MILLION','BLACK','HARD','STRONG','VARIOUS',
'BELIEVE','PLAY','TYPE','SURFACE','VALUE',
'SOON','MEAN','NEAR','MODERN','TABLE',
'PEACE','RED','ROAD','TAX','SITUATION',
'PERSONAL','PROCESS','ALONE','GONE','NOR',
'IDEA','WOMEN','ENGLISH','INCREASE','LIVING',
'LONGER','BOOK','CUT','FINALLY','NATURE',
'PRIVATE','SECRETARY','THIRD','SECTION','CALL',
'FIRE','KEPT','GROUND','VIEW','DARK',
'PRESSURE','BASIS','SPACE','FATHER','EAST',
'SPIRIT','UNION','EXCEPT','COMPLETE','WROTE',
'RETURN','SUPPORT','ATTENTION','LATE','PARTICULAR',
'RECENT','HOPE','LIVE','ELSE','BROWN',
'BEYOND','PERSON','COMING','DEAD','INSIDE',
'REPORT','LOW','STAGE','MATERIAL','INSTEAD',
'READ','HEART','LOST','DATA','AMOUNT',
'PAY','SINGLE','COLD','MOVE','HUNDRED',
'RESEARCH','BASIC','INDUSTRY','TRIED','HOLD',
'COMMITTEE','ISLAND','EQUIPMENT','DEFENSE','ACTUALLY',
'SON','SHOWN','TEN','RIVER','RELIGIOUS',
'SORT','CENTRAL','DOING','REST','INDEED',
'CARE','PICTURE','DIFFICULT','SIMPLE','FINE',
'SUBJECT','RANGE','WALL','MEETING','FLOOR',
'BRING','FOREIGN','CENT','PAPER','SIMILAR',
'FINAL','NATURAL','PROPERTY','COUNTY','MARKET',
'POLICE','GROWTH','INTERNATIONAL','START','TALK',
'WRITTEN','STORY','HEAR','ANSWER','NEEDS',
'HALL','ISSUE','CONGRESS','WORKING','LIKELY',
'EARTH','SAT','PURPOSE','LABOR','STAND',
'MEET','DIFFERENCE','HAIR','PRODUCTION','FOOD',
'FALL','STOCK','WHOM','SENT','LETTER',
'PAID','CLUB','KNOWLEDGE','HOUR','YES',
'CHRISTIAN','SQUARE','READY','BLUE','BILL',
'TRADE','INDUSTRIAL','DEAL','BAD','MORAL',
'DUE','ADDITION','METHOD','NEITHER','THROUGHOUT',
'COLOR','TRY','ANYONE','READING','LAY',
'NATION','FRENCH','REMEMBER','SIZE','PHYSICAL',
'UNDERSTAND','RECORD','WESTERN','MEMBER','SOUTHERN',
'NORMAL','STRENGTH','POPULATION','VOLUME','DISTRICT',
'TEMPERATURE','TROUBLE','SUMMER','MAYBE','RAN',
'TRIAL','LIST','FRIEND','EVENING','LITERATURE',
'LED','MET','ARMY','ASSOCIATION','INFLUENCE',
'CHANCE','HUSBAND','STEP','FORMER','SCIENCE',
'STUDENT','CAUSE','MONTH','HOT','AVERAGE',
'SERIES','AID','DIRECT','WRONG','LEAD',
'PIECE','MYSELF','THEORY','SOVIET','ASK',
'FREEDOM','BEAUTIFUL','MEANING','FEAR','NOTE',
'LOT','SPRING','CONSIDER','BED','PRESS',
'ORGANIZATION','TRUTH','HOTEL','EASY','WIDE',
'DEGREE','HERSELF','RESPECT','FARM','PLANT',
'MANNER','REACTION','APPROACH','RUNNING','LOWER',
'GAME','FEED','COUPLE','CHARGE','EYE',
'DAILY','PERFORMANCE','BLOOD','RADIO','STOP',
'TECHNICAL','PROGRESS','ADDITIONAL','MARCH','MAIN',
'CHIEF','WINDOW','DECISION','RELIGION','TEST',
'IMAGE','CHARACTER','MIDDLE','APPEAR','BRITISH',
'RESPONSIBILITY','GUN','LEARNED','HORSE','ACCOUNT',
'WRITING','SERIOUS','LENGTH','GREEN','ACTIVITY',
'FISCAL','CORNER','FORWARD','HIT','AUDIENCE',
'SPECIFIC','NUCLEAR','DOUBT','STRAIGHT','LATTER',
'QUALITY','JUSTICE','DESIGN','PLANE','SEVEN',
'STAY','POOR','BORN','CHOICE','OPERATION',
'PATTERN','STAFF','FUNCTION','INCLUDE','WHATEVER',
'SUN','SHOT','FAITH','POOL','WISH',
'LACK','SPEAK','HEAVY','MASS','HOSPITAL',
'BALL','STANDARD','AHEAD','VISIT','DEEP',
'LANGUAGE','FIRM','PRINCIPLE','CORPS','INCOME',
'DEMOCRATIC','NONE','EXPECT','DISTANCE','IMPORTANCE',
'PRICE','ANALYSIS','SERVE','PRETTY','ATTITUDE',
'CONTINUE','DETERMINE','EXISTENCE','DIVISION','STRESS',
'HARDLY','WRITE','SCENE','REACH','LIMITED',
'APPLIED','AFTERNOON','DRIVE','PROFESSIONAL','STATION',
'HEALTH','ATTACK','SEASON','SPENT','EIGHT',
'ROLE','CURRENT','NEGRO','ORIGINAL','BUILT',
'DATE','MOUTH','RACE','UNIT','TEETH',
'MACHINE','COUNCIL','COMMISSION','NEWS','SUPPLY',
'RISE','DEMAND','UNLESS','BIT','SUNDAY',
'OFFICER','MEANT','WALK','DOCTOR','ACTUAL',
'CLAY','GLASS','POET','JAZZ','CAUGHT',
'HAPPY','FIGHT','POPULAR','CONCERN','SHARE',
'STYLE','BRIDGE','GAS','CLAIM','FOLLOW',
'THOUSAND','SUPPOSE','HEAT','STATUS','CHRIST',
'CATTLE','RADIATION','USUAL','FILM','OPINION',
'PRIMARY','BEHAVIOR','CONFERENCE','SEA','PROPER',
'ATTEMPT','MARRIAGE','SIR','HELL','CONSTRUCTION',
'WORTH','PRACTICE','SIGN','SOURCE','WAIT',
'ARM','PARK','TRADITION','REMAIN','PROJECT',
'AUTHORITY','LORD','ANNUAL','JUNE','OIL',
'OBVIOUS','THIN','FELL','PRINCIPAL','JACK',
'CONDITION','DINNER','BASE','STRUCTURE','MEASURE',
'WEIGHT','OBJECTIVE','CIVIL','COMPLEX','MANAGEMENT',
'MIKE','EQUAL','NOTED','KITCHEN','DANCE',
'BALANCE','CORPORATION','PASS','FAMOUS','REGARD',
'DEVELOP','FAILURE','CLOTHES','COVER','BREAK',
'CARRY','MOREOVER','KEY','KING','ADD',
'ACTIVE','CHECK','BOTTOM','PAIN','MANAGER',
'ENEMY','POETRY','TOUCH','FIXED','POSSIBILITY',
'SPOKE','BRIGHT','BATTLE','PRODUCT','BUILD',
'SIGHT','ROSE','LOSS','PREVIOUS','FINANCIAL',
'PHILOSOPHY','REQUIRE','SCIENTIFIC','SHAPE','MARKED',
'MUSICAL','VARIETY','GERMAN','CAPITAL','CAPTAIN',
'CONCEPT','DISTRIBUTION','IMPOSSIBLE','LEARN','BEGIN',
'AWARE','BROAD','STRANGE','SEX','POST',
'CATHOLIC','REGULAR','OPENING','WINTER','CAPACITY',
'SHIP','SPREAD','HOUSES','PREVENT','MARK',
'SPEED','YESTERDAY','TEAM','BANK','GOVERNOR',
'INSTANCE','TRAIN','YOUTH','PRODUCE','FRESH',
'CRISIS','BAR','DRINK','IMMEDIATE','ROUND',
'WATCH','LIVES','ESSENTIAL','TRIP','NINE',
'EVENT','APARTMENT','CAMPAIGN','FILE','OPPOSITE',
'NECK','INDEX','TWENTY','OFFER','GRAY',
'LADY','FULLY','INDICATE','SESSION','RUSSIAN',
'PROVIDENCE','STUDIED','SEPARATE','ATMOSPHERE','PROCEDURE',
'TERM','EXPRESSION','REALITY','MAXIMUM','ECONOMY',
'SECRET','MISSION','FAST','FAVOR','EDGE',
'TONE','ENTER','LITERARY','COFFEE','SOLID',
'LAID','FAIR','PERMIT','RESPONSE','TITLE',
'JUDGE','ADDRESS','MODEL','ELECTION','ANODE'
]
