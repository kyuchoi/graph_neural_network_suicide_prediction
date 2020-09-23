# install.packages('bootnet')
library(bootnet)
# install.packages('qgraph')
library(qgraph)
library(dplyr) # for select
#library(rstudioapi)
# install.packages("here")
library(here)

### ref tutorial: http://sachaepskamp.com/files/Cookbook.html#network-estimation-binary-data

current_path <- here()
root_dir <- dirname(dirname(current_path)) # double dirname when you run in RStudio, but single dirname when you run with Rscript on command lines
# current_path <- getActiveDocumentContext()
# root_dir <- dirname(dirname(current_path$path))
raw_data_dir <- file.path(root_dir, 'raw_data') 
result_dir <- file.path(root_dir, 'result') 

if (!dir.exists(result_dir)){
  dir.create(result_dir, showWarnings = FALSE)
}
  
data_dir <- file.path(raw_data_dir, 'raws_for_MaDE_pseudo') 

label_name <- 'MaDE' # 'suicidal_idea' # 

filepath <- file.path(data_dir, 'df_test_KAIST.csv')
df <- read.csv(filepath)

list_feat <- c('PHQ_', 'GAD_', 'STAI_') #, 'site' is all the same as 1 for 'MaDE' 

df <- select(df, starts_with(list_feat))
df.sub <- df[1:nrow(df),]

####### compute correlation matrix from dataframe
df.sub.cor <- cor_auto(df.sub)
# heatmap(df.sub.cor) # check for if questionnaires cluster

### added threshold = TRUE for higher specifity at the cost of sensitivity, following error msgs.
Graph_lasso <- qgraph(df.sub.cor, graph = "glasso", layout = "spring", tuning = 0.25,
                      sampleSize = nrow(df.sub), filename= file.path(result_dir, 'qgraph_MaDE_result'), filetype='jpg')#, threshold = TRUE) # too low sensitivity ??

adj_from <- Graph_lasso$Edgelist$from
adj_to <- Graph_lasso$Edgelist$to
adj_weight <- Graph_lasso$Edgelist$weight

filename_feat <- paste0(list_feat, collapse = '_')

setwd(raw_data_dir)

write.csv(adj_from, paste0('df_from_',label_name,'.csv'))
write.csv(adj_to, paste0('df_to_',label_name,'.csv'))
write.csv(adj_weight, paste0('df_weight_',label_name,'.csv'))
