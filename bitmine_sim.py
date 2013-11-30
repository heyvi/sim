import math

def getTimePerBlock(difficulty,hashrate):
    target = 0x00000000ffff0000000000000000000000000000000000000000000000000000 / difficulty
    return math.pow(2,256)/(target*hashrate)

def calculate(hashrate,difficulty=None,exchange_rate=None):
    if difficulty is None: difficulty = 1
    if exchange_rate is None: exchange_rate = 1
    
    time_per_block = getTimePerBlock(difficulty,hashrate)
    coins = 25 / time_per_block
    dollars = coins * exchange_rate
    return dollars

def getHumanHashRate(hashrate):
	#this hash rate is straigh up hashes/sec, no metric prefix
    suffix = ['','K','M','G','T','P']
    humanhashrate = hashrate
    factor = 0
    while humanhashrate > 1000 and factor < len(suffix):
        humanhashrate /= 1000.0
        factor += 1
    return "{0:.2f}{1}".format(humanhashrate,suffix[factor])

def convertHashRate(humanHash):
	#human hash rate is set to be in Ghash/s
	hashrate = humanHash * 1000 * 1000 * 1000;
	return hashrate;


exchange_rate = 1000
base_difficulty = 609482679.888
humanHash = 600 # hashrate in GH/s
convertedHash = convertHashRate(humanHash)
difficulty_growthrate = .2

#delayed difficulty is to take into account changes in difficulty
#due to delays experienced while waiting for your mining hardware

delayed_difficulty = base_difficulty
delayed_weeks = 8 #set to 0 f you don't want a delayed difficulty
numWeeks = 52-8

for i in range(delayed_weeks/2):
	delayed_difficulty = delayed_difficulty * (1 + difficulty_growthrate)

running_difficulty = delayed_difficulty
running_total = 0

for i in range(numWeeks/2):
	dpSec = calculate(convertedHash, running_difficulty, exchange_rate)
	dpTwoWeeks = dpSec * 60 * 60 * 24 * 14
	running_total += dpTwoWeeks;
	running_difficulty = running_difficulty * (1 + difficulty_growthrate)

print(running_total);