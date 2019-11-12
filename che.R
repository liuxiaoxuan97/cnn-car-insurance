df<-read.table("C:/Users/lenovo/Desktop/freq_age_group.txt",header=TRUE,sep=",")
as.data.frame(df)
names(df)
names(df1)
df1<-df[,21:23]
dim(df1)
df2<-read.table("C:/Users/lenovo/Desktop/heatmap.txt",sep=",")
class(df2)
colnames(df2)
df2$Device_ID<-df2$V1

df3<-merge(df2,df1,by="Device_ID",all=FALSE)
dim(df3)
df3<-df3[,-2]
data<-df3[1:1478,]
df3$pc<-df3$Counts/df3$YEARS
summary(df3$pc)
data0<-subset(df3,df3$pc>0.4|df3$pc==0)



dim(data0)
data1<-df3[order(-df3$pc),]
data1$pc[365]
data2<-data1[366:1478,]
dim(data3)
data3<-data2[sample(1:nrow(data2),1113,replace=FALSE),]
#将ID 与cnn输出联系在一起，从而进行glm合并
ID<-data3[951:1113,1]
pre<-read.table("C:/Users/lenovo/Desktop/pre2.txt",sep=" ")
pre<-as.data.frame(pre)
dim(pre)
dm1<-data.frame(cbind(ID,pre))
unique(dm1$pre1)
pc<-data3$pc[951:1113]
colnames(dm1)<-c("Device_ID","pre1","pre2")
names(dm1)

#利用id将age gender years和count carage联系起来
unique(dglm1$AGE_Group)
dglm<-subset(df,select=c("SEX","USEYEARS","Device_ID","Counts","AGE_Group"))
dglm$YEARS<-df$YEARS
length(dglm$YEARS)
dglm1<-merge(dglm,dm1,by="Device_ID")
length(dm1)
dglm1
dglm1$SEX<-as.factor(dglm1$SEX)
dglm1$SEX<-dglm$SEX
dglm1$AGE_Group<-as.factor(dglm1$AGE_Group)
class(dglm1$SEX)
dglm1$USEYEARS
mod1<-glm(Counts~AGE_Group+SEX+USEYEARS+pre1+pre2,family = poisson(link="log"),data=dglm1)
mod2<-glm(Counts~AGE_Group+SEX+USEYEARS,family = poisson(link="log"),data=dglm1)
summary(mod1)
summary(mod1)$dispersion
deviance(mod2)
deviance(mod1)
anova(mod2)
summary(mod1)$dispersion
dglm1<-within(dglm1,{
  Counts0<-NA
  Counts0[dglm1$Counts==0]<-0
  Counts0[dglm1$Counts>0]<-1
})
dglm1$Counts0<-as.factor(dglm1$Counts0)
summary(dglm1$Counts0)
mod1<-glm(Counts0~AGE_Group+SEX+USEYEARS+pre1+pre2,family = binomial(link="logit"),data=dglm1)
summary(mod1)
#计算logistic ROC
library(pROC)
roc(dglm1$Counts0,predict(mod1,dglm1,type="response"))$auc
plot(roc(dglm1$Counts0,predict(mod1,dglm1,type="response")),print.thres=TRUE,print.auc=TRUE,col="red")
mod2<-glm(Counts0~AGE_Group+SEX+USEYEARS,family = binomial(link="logit"),data=dglm1)
roc(dglm1$Counts0,predict(mod2,dglm1,type="response"))$auc
plot(roc(dglm1$Counts0,predict(mod2,dglm1,type="response")),print.thres=TRUE,print.auc=TRUE,col="red")

mod1$
#生成label列

label<-data.frame(data3$Device_ID,data3$pc)

names(label)
c<-as.vector(data3$pc)
write.csv(c,"C:/Users/lenovo/Desktop/c600.csv")
#shengcheng shuzu
data4<-data3[,21:620]
dim(data4)
c1<-rowSums(data4)
c1<-as.vector(c1)
data4$sum<-c1


c1<-data4[2,2:600]

dat<-as.matrix(data4)
datNormed <- dat/dat[,601]
datNormed[1]

#d1是百分比后的矩阵
d1<-as.data.frame(datNormed)
d1<-d1[,-601]
dim(d1)
write.csv(d1,"C:/Users/lenovo/Desktop/cc600.csv")


#生成label列

label<-data.frame(df3$Device_ID,df3$pc)

names(label)
c<-as.vector(df3$pc)
write.csv(c,"C:/Users/lenovo/Desktop/c562.csv")
#shengcheng shuzu
data00<-df3[,81:640]
dim(data00)
c1<-rowSums(data00)
c1<-as.vector(c1)
data00$sum<-c1


c1<-data00[2,2:560]

dat<-as.matrix(data00)
datNormed <- dat/dat[,561]
datNormed[1]
#d1是百分比后的矩阵
d1<-as.data.frame(datNormed)
d1<-d1[,-561]
dim(d1)
write.csv(d1,"C:/Users/lenovo/Desktop/cc562.csv")

ht1<-read.csv("C:/Users/lenovo/Desktop/lxx/cc55.csv")
g<-as.numeric(ht1[33,])
g2<-as.data.frame((matrix(g,ncol=16)))
g3<-g2[1:20,]
Heatmap(as.matrix(g3),Rowv = NA,Colv = NA,col=heat.colors(800),scale="column",margins = c(1,8),show_row_names=FALSE)

library(RColorBrewer)

g1<-as.numeric(ht1[1159,])
g4<-as.data.frame((matrix(g1,ncol=16)))
g5<-g4[1:20,]

if(!require("devtools")) install.packages("devtools") 
devtools::install_github("jokergoo/ComplexHeatmap")
install.packages("pheatmap")
library(pheatmap)
library(gplots)
heatmap.2(as.matrix(g3))
pheatmap(g3,color=colorRampPalette(c("black","white"))(50) ,cluster_row=FALSE,fontsize = 9,fontsize_row = 6,)
heatmap(as.matrix(g5),Rowv = NA,Colv = NA,col=heat.colors(250),scale="column",labRow = rownames,labCol = FALSE)
rownames<-c(-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8)
rug(jitter(as.matrix(g3),amount=0.1))
scatterplot(g3)
g1<-as.numeric(ht1[1159,])
g4<-as.data.frame((matrix(g1,ncol=16)))
g5<-g4[1:20,]
class(g5[1,1])
x<-seq(5,24,1)
y<-seq(-2,2,0.2)
filled.contour(x,y,g5,color=terrain.colors(3))
filled.contour(volcano,color=terrain.colors,asp=1)
class(volcano)
volcano[1,4]

library(FarmSelect)
library(tidyverse)
data1<-read.csv('C:/Users/lenovo/Desktop/2019-05.csv')
View(data2)
dim(data1)
data2<-na.omit(data1)
data1[0,]
sum(is.na(data1))
data1<-as.numeric(data1)
class(data1[2,2])
X=matrix(data2)
data2[324,3]
dim(X)
X[2]
plot(data2[,2:126])

output=farm.res(data2[,2:126],K.factors = 10)
dim(output$X.res)
View(data1)
install.packages("midas")
library(midasr)
diff(1:10, 2)
diff(1:10, 2, 2)
x <- cumsum(cumsum(1:10))
cumsum(1:10)
x
diff(x)
diff(x, differences = 2)
diff(.leap.seconds)
lws_table
library(MASS)
library(numDeriv)
library(Matrix)
install.packages("forecast")
library(forecast)
library(zoo)
library(stats)
library(graphics)
library(utils)
library(Formula)
install.packages('midasr')
library(texreg)
library(methods)
nlmn <- expand_weights_lags("nealmon",0,c(4,8),1,start=list(nealmon=rep(0,3)))
nbt <- expand_weights_lags("nbeta",0,c(4,8),1,start=list(nbeta=rep(0,4)))
nlmn+nbt
data("USunempr")
data("USrealgdp")
y <- diff(log(USrealgdp))
x <- window(diff(USunempr),start=1949)
x
trend <- 1:length(y)
tb <- amidas_table(y~trend+fmls(x,12,12,nealmon),
                   data=list(y=y,x=x,trend=trend),
                   weights=c("nealmon"),wstart=list(nealmon=c(0,0,0)),
                   start=list(trend=1),type=c("A"),
                   from=0,to=c(1,2))
y <- diff(log(USrealgdp))
x <- window(diff(USunempr),start=1949)
t <- 1:length(y)
fmls(x,11,12,nealmon)
mr <- midas_r(y~t+fmls(x,11,12,nealmon),start=list(x=c(0,0,0)))
mr$fitted.values
x<-1:16
mls(x,0:3,4)
