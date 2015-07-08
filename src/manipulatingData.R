file_list <- list.files("~/Dropbox/Summer 2015/ML/ML-4641-Project/src/data/")

for(i in 1:length(file_list)) {
  filepath <- file.path("~/Dropbox/Summer 2015/ML/ML-4641-Project/src/data/", paste(file_list[i],sep=""))
#   datafile <- read.csv(filepath)
  
  datafile <- read.csv(filepath)
  names(datafile) <- c("num", "open", "high", "low", "close", "volume", "adjusted")
  datafile$volume <- as.numeric(datafile$close) - as.numeric(datafile$open)
  datafile$num[1]=0
  for(j in 2:length(datafile$adjusted)) {
      datafile$num[j] <- ((datafile$adjusted[j]-datafile$adjusted[j-1])/datafile$adjusted[j-1])*100
  }
  names(datafile) <- c("% change", "open", "high", "low", "close", "variation", "adjusted")
  datafile$open <- NULL
  datafile$close <- NULL
  datafile$high <- NULL
  datafile$low <- NULL
  ticker <- gsub("\\..*","",file_list[i])
  datafile$ticker <- ticker
  temp <- merge(temp, datafile)
  write.csv(datafile, file=file_list[i])
  datafile<- NULL
}
