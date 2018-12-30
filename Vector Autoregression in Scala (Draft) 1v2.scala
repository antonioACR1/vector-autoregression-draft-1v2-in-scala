import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import sqlContext.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics._
import org.apache.commons.math3.stat.regression.AbstractMultipleLinearRegression
import scala.util.{Try,Success,Failure}
import breeze.linalg.DenseMatrix
import breeze.linalg.diag
import Array._ /*to define arrays of consecutive values by using range(a,b)*/
import scala.math.pow 


class varModel(df:Array[Array[Double]],exogen:Option[Array[Array[Double]]],interceptTrend:Option[String], criteria : String, pValue:Int,maxLag:Option[Int],season:Option[Int]) {
/*BASIC FUNCTIONS*/
/*this methods divides a range of consecutive integers of size sizeMat into K blocks and withdraws the first blockSize elements of each block*/
/*for example, jumping(9,1,3) withdraws from Array(0,1,2,3,4,5,6,7,8) returns */
def jumping(sizeMat : Int,blockSize:Int,K:Int):List[Int]={
  /*the arrays inside df should be non-empty and should have the same size*/
  val jump=sizeMat/K
  var g=Array[Int]()
  for (i<-0 to K-1){val s=(jump*i to jump*i+blockSize).toList
    g=g++s
  }
  val g1=g.toList
  g1
}

/*this a slight variation of lagMatTrimBoth in order to get the full matrix of shifts from a time series (the equivalent of embed() in R)*/

def fixedlagMatTrimBothArr(dfArrCollect:Array[Array[Double]],lag:Int):Array[Array[Double]]={
  val incompleteShifts=Lag.lagMatTrimBoth(dfArrCollect,lag)
  val K : Int =dfArrCollect(0).size
  val nShifts : Int =(incompleteShifts(0).size.toInt/K).toInt
  val lastRow=dfArrCollect.drop(dfArrCollect.size-nShifts)
  val lastRowArr= lastRow.map{r=>val
    array=r.toSeq.toArray
    array.map(_.asInstanceOf[Double])
  }
  val lastRowDM=DenseMatrix(lastRowArr.map(_.toArray):_*)
  var lastArr=Array[Double]()
  for (i<-0 to lastRow(0).size-1){
    val col=lastRowDM(::,i).toArray
    /*change the following lines by (0 to col.size-1).map(i=>col.size-1-i).map(i=>col(i))*/
    var colVar=Array[Double]() 
      for (j<-0 to col.size-1){
        colVar=colVar:+col(col.size-1-j)
      }
    lastArr=lastArr++colVar 
  }
  val completeShifts=incompleteShifts:+lastArr
  completeShifts
}

def Lasty(dfArr:Array[Array[Double]],p:Int):Array[Double]={
  val K=dfArr(0).size
  var lasty=Array[Double]()
   for(i<-0 to K-1){
     for(j<-1 to p){
       lasty=lasty:+dfArr(dfArr.size-j)(i)
     }
   }
  lasty
}
    /*method to convert an array of arrays to dense matrix*/
def toDM(Arr:Array[Array[Double]]):breeze.linalg.DenseMatrix[Double]={
    val DM=breeze.linalg.DenseMatrix(Arr.map(_.toArray):_*)
    DM
}
    /*method which constructs the columns corresponding to intercept, trend, both or none. This construction is denoted by RHS in R*/
def RHS(interceptTrend:String, initialIndex: Int,length: Int):breeze.linalg.DenseMatrix[Double]={
    val rhs=interceptTrend match {
    case "intercept" => DenseMatrix(Array.fill(length)(1.0)).t
    /*case "trend" => DenseMatrix(((initialIndex+1) to (initialIndex+length) by 1).toArray.map(_.toDouble)).t*/
    case "trend" => DenseMatrix(Array.range(initialIndex+1,initialIndex+length+1).map(_.toDouble)).t
    /*case "both" => DenseMatrix.horzcat(DenseMatrix(Array.fill(length)(1.0)).t,DenseMatrix(((initialIndex+1) to (initialIndex+length) by 1).toArray.map(_.toDouble)).t)*/
    case "both" => DenseMatrix.horzcat(DenseMatrix(Array.fill(length)(1.0)).t,DenseMatrix(Array.range(initialIndex+1,initialIndex+length+1).map(_.toDouble)).t)
    case _ => DenseMatrix(Array.fill(length)(0.0)).t /*dummy*/
    }
    rhs
}
    /*method to construct seasonal columns to be concatenated with RHS*/
def seasonality(season:Int,initialIndex:Int,finalIndex:Int,length:Int):breeze.linalg.DenseMatrix[Double]={
    val a=diag(DenseVector.fill(season){1.0}) /*diagonal matrix*/
    val b=new DenseMatrix(season,season,DenseVector.fill(season*season){1.0}.toArray)*(1/season.toDouble) /*substract 1/season entry-wise*/
    val c=a-b
    var cVar=c
    while(cVar(::,0).size <= length){ /*concat until length size*/
    cVar=DenseMatrix.vertcat(cVar,c);
    }
    cVar=cVar(initialIndex to finalIndex,0 to season-2) /*cut*/
    cVar
}
/*exogenous option*/

def getExogen(exogenArray:Array[Array[Double]],initialIndex:Int):breeze.linalg.DenseMatrix[Double]={
    val exogenDM=toDM(exogenArray.drop(initialIndex))
    exogenDM
}
def RegressionErrors(y:Array[Double],X:Array[Array[Double]]):Array[Double]={
    val regression=new OLSMultipleLinearRegression()
    regression.setNoIntercept(true)
    val params=Try{regression.newSampleData(y,X)} match {case
        Success(_)=>regression.estimateResiduals() 
        case Failure(_)=>Array.fill(X(0).size)(0.0)
        }
    params
}
def RegressionCoefficients(y:Array[Double],X:Array[Array[Double]]):Array[Double]={
    val regression=new OLSMultipleLinearRegression()
    regression.setNoIntercept(true)
    regression.newSampleData(y,X) 
    val params=regression.estimateRegressionParameters()
    params
}
def criteriaLagSearch(ic:String,coeffMatrix:breeze.linalg.DenseMatrix[Double],K:Int,lengthShift:Int,detint:Int,step:Int,numberColumnsPerBlock:Array[Int]):Double={
    val detSigma=breeze.linalg.det((coeffMatrix*coeffMatrix.t):*(1/lengthShift.toDouble)) 
    val stepValue=ic match {
        case "FPE" => scala.math.pow((lengthShift + numberColumnsPerBlock(step)) / (lengthShift - numberColumnsPerBlock(step)),K) * detSigma
        case "HQ" => breeze.numerics.log(detSigma) + (2*breeze.numerics.log(breeze.numerics.log(lengthShift))/lengthShift)*((step+1)*scala.math.pow(K,2)+K*detint)
        case "SC" => breeze.numerics.log(detSigma) + (breeze.numerics.log(lengthShift)/lengthShift)*((step+1)*K*K+K*detint)
        case "AIC" => breeze.numerics.log(detSigma)+(2/lengthShift)*((step+1)*K*K+K*detint)
    }
    stepValue
}

def flatTuple(arrayTuple:Array[(Int,Array[Double])]):Array[Array[Double]]={
val myFlattenArray=arrayTuple.map{case (id,array)=>array}
myFlattenArray 
}

def unionOriginalAndForecast(dfArr:Array[Array[Double]],forecast:Array[Double]):Array[Array[Double]]={
  val dfArrayUnion=dfArr:+forecast
  dfArrayUnion
}
/*--------------------------------------------------------------------------------------*/

    /*WITHDRAW INITIAL VARIABLES*/
    val K : Int = df(0).size
    val nObs : Int = df.size
    val interceptTrendChoice=interceptTrend.getOrElse("none")
    val exogenChoice=exogen.getOrElse(Array(Array(0.0)))
    val maxLagChoice=maxLag.getOrElse(0)
    val seasonChoice=season.getOrElse(0)
    val pChoice=pValue
    val ic=criteria

/*Roughly speaking, the idea of the following method is to construct a big matrix (yLaggedDM) and then compute several linear regressions extracting features from this matrix, the getting the coefficients of each regression model and then apply the formula of the correspondig criteria to get the appropriate lag. Intuitively, lag refers to how many steps back we apply linear regression on the time series itself*/

def lagSelection(maxLagChoice:Int) : (Int,Array[Double])={
  val yendogArr=df.drop(maxLagChoice)
  val yendog=toDM(yendogArr)   
  val lag=maxLagChoice+1
  val ylaggedFull=fixedlagMatTrimBothArr(df,lag)
  val ylaggedDM=toDM(ylaggedFull)
  val lengthShift=ylaggedFull.size
  var rhs=RHS(interceptTrendChoice,maxLagChoice,lengthShift) 
  if(seasonChoice!=0){
  val seasonBlock=seasonality(seasonChoice,0,lengthShift-1,lengthShift)
  rhs=DenseMatrix.horzcat(rhs,seasonBlock)  
}
  if(exogenChoice!=Array(Array(0.0))){
  val exogenBlock=getExogen(exogenChoice,maxLagChoice)
  rhs=DenseMatrix.horzcat(rhs,exogenBlock)
}
  if(interceptTrendChoice=="none"){
  rhs=rhs(::,1 to rhs.cols-1)
}

/*construct source matrix from which we will withdraw the features and labels for our set of k linear regressions, each set having either 1, 2, ..., maxLag features per variable. The idea is to broadcast this source matrix to each node later on*/

val source=DenseMatrix.horzcat(ylaggedDM,rhs)
val detint : Int = rhs.cols
val numberColumnsPerBlock=Array.range(1,lag).map(i=>i*K+detint)
val rhsIndexes=Array.range(lag*K,lag*K+detint).toList

/*construct features and labels by indexes and by blocks*/
val featuresIndexesByBlock=Array.range(1,lag).map(i=>jumping(lag*K,i,K).filterNot(jumping(lag*K,0,K).contains(_))).map(i=>i++rhsIndexes)
val labelsIndexes=jumping(lag*K,0,K)

val labelsFeaturesIndexes=labelsIndexes.flatMap(label=>featuresIndexesByBlock.map(features=>((features.size-detint)/K,label,features)))
val labelsFeatures=labelsFeaturesIndexes.map{case (id,label,features)=>(id,source(::,label),source(::,features))}.map{case (id,labelDV,featuresSM)=>(id,labelDV.toArray,MatrixUtil.matToRowArrs(featuresSM))}

val coefficientsById=Array.range(1,lag).flatMap(i=>labelsFeatures.filter{case (id,label,features)=>id==i}).map{case (id,label,features)=>(id,RegressionErrors(label,features))}
/*val matrices=coefficientsById.groupBy(_._1).map{case (key,value)=>flatTuple(value)}.toArray.map(i=>toDM(i))*/
/*Array(1,2).map(i=>Array((1,Array(1.0,2.0)),(1,Array(3.0,4.0)),(2,Array(11.0,12.0)),(2,Array(13.0,14.0))).filter{case (a,b)=>a==i}).map{i=>toDM(flatTuple(i))}*/
val matrices=Array.range(1,lag).map(i=>coefficientsById.filter{case (a,b)=>a==i}).map{i=>toDM(flatTuple(i))}

val criteriaValues=Array.range(0,lag-1).map(i=>criteriaLagSearch(ic,matrices(i),K,lengthShift,detint,i,numberColumnsPerBlock))
val selectedLag=criteriaValues.zipWithIndex.min._2.toInt + 1
(selectedLag,criteriaValues)
}

        /*set lag depending on user's choice*/
val p=if(maxLagChoice!=0){lagSelection(maxLagChoice)._1} else {pChoice}

/*----------------------------------------------------------------*/
/*Getting matrix of coefficients*/
/*-----------------------------------------------------------------*/

  val yendogArr=df.drop(p)
  val lag=p+1
  val ylaggedFull=fixedlagMatTrimBothArr(df,lag)
  val ylaggedDM=toDM(ylaggedFull)
  val lengthShift=ylaggedFull.size
  var rhs=RHS(interceptTrendChoice,p,lengthShift)
  if(seasonChoice!=0){
  val seasonBlock=seasonality(seasonChoice,p,nObs-1,lengthShift)
  rhs=DenseMatrix.horzcat(rhs,seasonBlock)  
  }
  if(exogenChoice!=Array(Array(0.0))){
  val exogenBlock=getExogen(exogenChoice,p)
  rhs=DenseMatrix.horzcat(rhs,exogenBlock)
  }
  if(interceptTrendChoice=="none"){
    rhs=rhs(::,1 to rhs.cols-1)
  }
  val source=DenseMatrix.horzcat(ylaggedDM,rhs)
  val detint : Int = rhs.cols
  val rhsIndexes=Array.range(lag*K,lag*K+detint).toList

  /*construct features and labels by indexes and by blocks*/
  val featuresIndexes=jumping(lag*K,p,K).filterNot(jumping(lag*K,0,K).contains(_))++rhsIndexes
  val labelsIndexes=jumping(lag*K,0,K)

  val labelsFeaturesIndexes=labelsIndexes.map(label=>(label,featuresIndexes))
  val labelsFeatures=labelsFeaturesIndexes.map{case (label,features)=>(source(::,label),source(::,features))}.map{case (labelDV,featuresSM)=>(labelDV.toArray,MatrixUtil.matToRowArrs(featuresSM))}

  val coefficients=labelsFeatures.map{case (label,features)=>(RegressionCoefficients(label,features))}.toArray
  val matrix=toDM(coefficients)

/*---------------------------------------------------------------------------*/
/*Getting errors*/
/*----------------------------------------------------------------------------*/
  val nAhead=nObs-p
  var ZdetDM=RHS(interceptTrendChoice,p,nAhead)
  if(seasonChoice!=0){
      val seasonBlock=seasonality(seasonChoice,p,nObs-1,nObs-p)
      ZdetDM=DenseMatrix.horzcat(ZdetDM,seasonBlock)  
  }
  if(exogenChoice!=Array(Array(0.0))){
      val exogenBlock=getExogen(exogenChoice,p)
      ZdetDM=DenseMatrix.horzcat(ZdetDM,exogenBlock)
  }
if(interceptTrendChoice=="none"){
  ZdetDM=ZdetDM(::,1 to ZdetDM.cols-1)
}
  val ZdetArr=MatrixUtil.matToRowArrs(ZdetDM)
  val LHSparts=Lasty(df.take(p),p)+:Array.range(0,nAhead-1).map(i=>Lasty(df.take(p+i+1),p))
  val RHSparts=Array.range(0,nAhead).map(i=>ZdetArr(i))
  val predictions=Array.range(0,nAhead).map(i=>DenseMatrix(LHSparts(i)++RHSparts(i))*matrix.t).map(i=>i.toArray)
  /*getting errors*/
  val actualValues=df.drop(p)
  var errors=Array[Array[Double]]()
  for (c<-0 to K-1){
    val a=predictions.map(x=>x(c))
    val b=actualValues.map(x=>x(c))
    val error=a.zip(b).map {case (x,y)=>y-x}
    errors=errors:+error
  }
  val residuals=toDM(errors).t
/*---------------------------------------------------------------------------------*/
/*PREDICT*/
/*---------------------------------------------------------------------------------*/


def predict(newExogen:Option[Array[Array[Double]]],nAheadNew:Int):breeze.linalg.DenseMatrix[Double]={
val newExogenChoice=newExogen.getOrElse(Array(Array(0.0)))
  /*construction of Zdet*/
  val trdstart = nObs+1
  var ZdetDMPredict=RHS(interceptTrendChoice,trdstart,nAheadNew)
    if(seasonChoice!=0){
    val cVar=seasonality(seasonChoice,p,nObs-1,nObs-p)
    val cycle=cVar(cVar.rows-seasonChoice to cVar.rows-1,::)
    var seasonal=cycle    
    while(seasonal.rows<=nAheadNew){
      seasonal=DenseMatrix.vertcat(seasonal,cycle)
    }
    seasonal=seasonal(0 to nAheadNew-1,::)
    ZdetDMPredict=DenseMatrix.horzcat(ZdetDMPredict,seasonal)
  }
    if(newExogenChoice!=Array(Array(0.0))){
      val newExogenBlock=getExogen(newExogenChoice,0)
      ZdetDMPredict=DenseMatrix.horzcat(ZdetDMPredict,newExogenBlock)
  }
  if(interceptTrendChoice=="none"){
    ZdetDMPredict=ZdetDMPredict(::,1 to rhs.cols-1)
 }
  val ZdetArrPredict=MatrixUtil.matToRowArrs(ZdetDMPredict)
  /*prepare first element for the loop*/
  var predictionsNew=Array[Array[Double]]()
  var lastyNew : Array[Double] =Lasty(df,p)
  /*prepare first dfArr*/
  var dfArrNew= df
  /*predictions using a loop*/
  for (i<-0 to (nAheadNew-1)){
    val ZArrNew=lastyNew++ZdetArrPredict(i)
    val Z=DenseMatrix(ZArrNew)
    val forecastDM=matrix*Z.t
    val forecastArr=MatrixUtil.matToRowArrs(forecastDM.t)(0)
    predictionsNew=predictionsNew :+ forecastArr
    dfArrNew=unionOriginalAndForecast(dfArrNew,forecastArr)
    lastyNew=Lasty(dfArrNew,p)
  }
  val predictionDM=toDM(predictionsNew)
  predictionDM
}
}


