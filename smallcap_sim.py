import ystockquote
import datetime

# Script to evaluate whether DCA is better than lump sum investing
# testing on the Russell 2000 small cap index fund

start_date = datetime.date(2001, 1, 1)
end_date = datetime.date(2001, 12, 31)
one_day = datetime.timedelta(days=1)
stock_data = ystockquote.get_historical_prices('^RUT', start_date.isoformat(),end_date.isoformat() );

def findStartDate(date):
	dateToTry = date

	while(True):
		try:
			stock_data[dateToTry.isoformat()]
			return dateToTry
		except:
			dateToTry = dateToTry + one_day
			continue


def findEndDate(date):
	dateToTry = date

	while(True):
		try:
			stock_data[dateToTry.isoformat()]
			return dateToTry
		except:
			dateToTry = dateToTry - one_day
			continue

#find valid start and end dates
start_date = findStartDate(start_date)
end_date = findEndDate(end_date)

investment_amount = 10000.0
lumpsum_purchase_price = float(stock_data[start_date.isoformat()]['Close'])
shares_bought_lumpsum = investment_amount/lumpsum_purchase_price

shares_bought_dca = 0
reinvest_gap = datetime.timedelta(days=15)
dca_invest_rate = investment_amount/((end_date-start_date).days/reinvest_gap.days)
dca_invest_rate *= 1.05 #correction factor, conservative in this case

dateProbe = start_date
num_investments = 0
amt_invested = 0
last_invest_date = start_date

while(dateProbe != end_date):

	try:
		daily_close = float(stock_data[dateProbe.isoformat()]['Close'])

		if(dateProbe > last_invest_date + reinvest_gap):
			num_investments += 1
			amt_invested += dca_invest_rate
			shares_bought_dca += dca_invest_rate/daily_close
			last_invest_date = dateProbe
		
		dateProbe = dateProbe + one_day

	except:
		dateProbe = dateProbe + one_day
		continue


current_price = float(stock_data[end_date.isoformat()]['Close'])
lumpsum_worth = shares_bought_lumpsum * current_price
dca_worth = shares_bought_dca * current_price

dca_remainder = investment_amount - amt_invested
dca_worth = dca_worth + dca_remainder

print("DCA STATS\nNum Investments : {0}\nAmt Invested:{1}".format(num_investments, amt_invested))
print "{0:2}: Shares Bought: {1:.2f}".format("Lumpsum", shares_bought_lumpsum)
print "{0:2}: Shares Bought: {1:.2f}".format("DCA", shares_bought_dca)
print "{0:2}: Investment Worth: {1:.2f}".format("Lumpsum", lumpsum_worth)
print "{0:2}: Investment Worth: {1:.2f}".format("DCA", dca_worth)

#Lumpsum was generally found to be better.

#Looking at data over a longer period from 1990 to 2013, we see that:
#Lump Sum: 67156
#DCA: 27263