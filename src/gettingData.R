# author : Bhavesh

# library whichis needed to get the data
require(quantmod)

# replace the directory to where your list file is located
stock_list <- read.csv("Desktop/sp500-symbol-list.txt",header = F, stringsAsFactors = F)

# print the list of S&P 500 tickers
stock_list

# create new environment to store the stock data
stock_data <- new.env()

# length of list <- 500
nrstocks = length(stock_list[,1])

# main loop which will download the data
for (i in 1:nrstocks) {
  tryCatch( {
    if (!is.null(eval(parse(text=paste("stock_data$",stock_list[i,1],sep = ""))))) {
      break
    }
    cat("(",i,"/",nrstocks,")","Downloading",stock_list[i,1],"\n")
    getSymbols(stock_list[i,1], env = stock_data, src="yahoo",from="2010-06-10", to="2015-06-10")
  }
  , error = function(e) print (e))
}

#print data for Apple stock
stock_data$AAPL
#save data to a csv file for further use
write.csv(stock_data$AAPL, file="apple.csv")


##########################################################################
# similary we would have to manually write csv files for all 500 stocks. 
# If we divide that, it wont take longer that half - one hour per person.