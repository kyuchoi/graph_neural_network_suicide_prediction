# install.packages('bootnet')
library(bootnet)
# install.packages('qgraph')
library(qgraph)
library(dplyr) # for select
# library(rstudioapi)
# library(lattice)
library(here)

### ref tutorial: http://sachaepskamp.com/files/Cookbook.html#network-estimation-binary-data

# get 'except_PHQ9' as an argument
# except.PHQ9 <- TRUE
args = commandArgs(trailingOnly=TRUE)
action = args[1] # action (i.e. --mean)
cat('processing:',action,'\n') # print out option
except.PHQ9 <- args[-1] # argument (i.e. kurtosis)
cat('except.PHQ9:',except.PHQ9,'\n')

current_path <- here() # getActiveDocumentContext()
root_dir <- dirname(current_path) # $path
setwd(root_dir)

data_dir <- file.path(root_dir, 'raw_data', 'data_for_smote')
label_name <- 'suicidal_idea_within_2wk'
filepath <- file.path(data_dir, 'df_train_valid.csv')
df <- read.csv(filepath)
# table(is.na(df)) # check for missing values: FALSE means not missing
idea <- df$'suicidal_idea_within_2wk'
list_feat <- c('Gender', 'site', 'PHQ', 'GAD', 'STAI_X1_total', 'RAS_total', 'RSES_total', 'MaDE', 'suicidal_attempt') # NEVER forget to use c('PHQ','GAD'), instead of just ('PHQ','GAD') # 
df <- select(df, starts_with(list_feat))
df.sub <- df[1:nrow(df),]

# remove PHQ_9 item which asks SI directly for ablation study
if (except.PHQ9){
  df.sub <- select(df.sub, -PHQ_9)
}

cor.PHQ9_STAI <- FALSE # TRUE #
cor.RAS_idea <- FALSE # TRUE # 
cor.RSES_idea <- FALSE # TRUE # 
use.EBIC <- FALSE # TRUE #

####### compute single correltion for PHQ9 and STAI: r=0.2 (p>0.001) -> weak positive corr
if (cor.PHQ9_STAI){
  RAS_total <- df.sub$RAS_total
  cor.test(RAS_total, idea)
} 

####### compute single correltion for RAS and idea: r=0.2 (p>0.001) -> weak positive corr
if (cor.RAS_idea){
  STAI_X1_total <- df.sub$STAI_X1_total
  cor.test(PHQ_9, STAI_X1_total)
} 

####### compute single correltion for RSES and idea: r=0.2 (p>0.001) -> weak positive corr
if (cor.RSES_idea){
  PHQ_9 <- df.sub$PHQ_9
  STAI_X1_total <- df.sub$STAI_X1_total
  cor.test(PHQ_9, STAI_X1_total)
} 
####### compute correlation matrix from dataframe
df.sub.cor <- cor_auto(df.sub)
heatmap(df.sub.cor) # check for if questionnaires cluster

lst.threshold.str <- 6 # c(4,5,6)

for (threshold.str in lst.threshold.str){
  threshold <- as.numeric(paste0('0.', threshold.str))
  
  if (use.EBIC){
    ### eventually, the same meaning as below line
    # EBICgraph <- EBICglasso(df.sub.cor, nrow(df.sub), 0.5, threshold = TRUE)
    # EBICgraph <- qgraph(EBICgraph, layout = "spring", title = "EBIC")
    
    ### added threshold = TRUE for higher specifity at the cost of sensitivity, following error msgs.
    Graph_lasso <- qgraph(df.sub.cor, graph = "glasso", layout = "spring", tuning = 0.5,
                          sampleSize = nrow(df.sub), threshold = TRUE) # too low sensitivity ?? setting threshold = True makes weights 163, and no threshold makes 198, which is consistent with the server model
    
  } else {
    
    ####### compute pairwise correlation matrix  
    ### ref document for qgraph: http://sachaepskamp.com/qgraph/reference/qgraph.html
    # threshold = 'bonferroni', 'hochberg','fdr'
    # cor for pairwise correlation, pcor for partial correlation 
    
    # res <- cor(as.matrix(df.sub)) 
    # res and df.sub.cor gives different results, because cor_auto automatically computes both pearson and spearman for appropriate data, but cor only gives pearson?
    Graph_lasso <- qgraph(df.sub.cor, graph = "cor", layout = "circle", palette = 'colorblind', threshold = threshold, sampleSize = nrow(df.sub),
                          filename = file.path(root_dir, 'qgraph_idea_result'), filetype='jpg')
    # Graph_lasso <- qgraph(df.sub.cor, graph = "cor", layout = "spring", threshold = 'hochberg', sampleSize = nrow(df.sub))
    ##### GIN performance depends on the qgraph threshold --> 0.4: e=197/0.5: e=161/0.6: e=127 (default)
  }
  
  adj_from <- Graph_lasso$Edgelist$from
  adj_to <- Graph_lasso$Edgelist$to
  adj_weight <- Graph_lasso$Edgelist$weight
  adj_weight.abs <- abs(adj_weight)
  
  filename_feat <- paste0(list_feat, collapse = '_')
  
  setwd(data_dir)
  
  ### save as csv
  if (except.PHQ9){
    write.csv(adj_from, paste0('df_from_',label_name,'_thr',threshold.str,'_x9.csv'))
    write.csv(adj_to, paste0('df_to_',label_name,'_thr',threshold.str,'_x9.csv'))
    #### write.csv(adj_weight, paste0('df_weight_',label_name,'_x9.csv'))
    write.csv(adj_weight.abs, paste0('df_weight_',label_name,'_thr',threshold.str,'_x9.csv'))
  }else{
    write.csv(adj_from, paste0('df_from_',label_name,'_thr',threshold.str,'.csv'))
    write.csv(adj_to, paste0('df_to_',label_name,'_thr',threshold.str,'.csv'))
    #### write.csv(adj_weight, paste0('df_weight_',label_name,'.csv'))
    write.csv(adj_weight.abs, paste0('df_weight_',label_name,'_thr',threshold.str,'.csv'))
  }
  
}

# install.packages('finalfit')
# library(finalfit)
