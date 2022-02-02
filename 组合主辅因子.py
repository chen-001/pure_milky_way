import numpy as np
import pandas as pd
import scipy.io as scio
import statsmodels.formula.api as smf
from pure_world import *



'''构造辅助因子部分'''
col=list(scio.loadmat('/Users/chenzongwei/pythoncode/数据库/日频数据/AllStockCode.mat').values())[3]
index=list(scio.loadmat('/Users/chenzongwei/pythoncode/数据库/日频数据/TradingDate_Daily.mat').values())[3]
col=[i[0] for i in col[0]]
index=[i[0] for i in index]

def read(path,col=col,index=index):
    path='/Users/chenzongwei/pythoncode/数据库/日频数据/'+path
    data=list(scio.loadmat(path).values())[3]
    data=pd.DataFrame(data,index=index,columns=col)
    data.index=pd.to_datetime(data.index,format='%Y%m%d')
    data=data.replace(0,np.nan)
    return data

tr=read('AllStock_DailyTR.mat')
close=read('AllStock_DailyClose_dividend.mat')
high=read('AllStock_DailyHigh_dividend.mat')
low=read('AllStock_DailyLow_dividend.mat')
open=read('AllStock_DailyOpen_dividend.mat')
close_lag=close.shift(1)

pct=(close*2-high-low)/close_lag
pct=pct.replace(-np.inf,np.nan)
pct=pct.replace(np.inf,np.nan)

fall=pure_fall()

'''两因子的月度值'''
tr_monthly_mean=tr.rolling(20).mean().resample('M').last()
tr_monthly_std=tr.rolling(20).std().resample('M').last()
pct_monthly=pct.rolling(20).mean().resample('M').last()
tr_monthly_neu=fall.get_neutral_monthly_factors(tr_monthly,boxcox=True)
pct_monthly_neu=fall.get_neutral_monthly_factors(pct_monthly,boxcox=True)
fall_tr_monthly_neu=pure_fallmount(tr_monthly_neu)
fall_pct_monthly_neu=pure_fallmount(pct_monthly_neu)


'''对比方案，两因子单测，换手率信息比率1.84，下影线减上影线信息比率1.70，换手率的标准差2.34'''###
shen_tr_monthly_neu=pure_moonnight(tr_monthly_neu,10,filename='原始换手率')
shen_pct_monthly_neu=pure_moonnight(pct_monthly_neu,10,filename='原始上下影线差')

'''组合方案一A，月频上标准化相加，信息比率2.75'''###
fall_plus_monthly=fall_tr_monthly_neu+(fall_pct_monthly_neu,)
shen_plus_monthly=pure_moonnight(fall_plus_monthly(),10,filename='月标_原始换手加原始影差')


# '''组合方案一B，月频上排序值相加，信息比率2.29'''
# fall_plus_monthly_rank=fall_tr_monthly_neu().rank(axis=1)+fall_pct_monthly_neu().rank(axis=1)
# shen_plus_monthly_rank=pure_moonnight(fall_plus_monthly_rank,10)

'''组合方案二A，月频上标准化归正相乘，信息比率2.76'''###
fall_prod_monthlyA=fall_tr_monthly_neu*(fall_pct_monthly_neu,)
shen_prod_monthly=pure_moonnight(fall_prod_monthlyA(),10,filename='月标_原始换手乘原始影差')

'''组合方案二B，月频上相乘，信息比率1.3'''###
fall_prod_monthlyB=tr_monthly_neu*pct_monthly_neu
shen_prod_monthly=pure_moonnight(fall_prod_monthlyB,10,filename='月_原始换手乘原始影差')

'''组合方案三预处理'''###
tr_daily_standard=fall.standardlize_in_cross_section(tr)
tr_daily_standard=(tr_daily_standard.T-tr_daily_standard.T.min()).T
pct_daily_standard=fall.standardlize_in_cross_section(pct)
pct_daily_standard=(pct_daily_standard.T-tr_daily_standard.T.min()).T
prod_daily_standardA=tr_daily_standard*pct_daily_standard

'''组合方案三A，日频上标准化归正相乘，月度取均值，信息比率0.11'''
prod_daily_to_monthly_standardA=prod_daily_standardA.rolling(20).mean().resample('M').last()
shen_prod_daily_standardA=pure_moonnight(prod_daily_to_monthly_standardA,10,boxcox=True,filename='日标_原始换手乘原始影差_mean')

'''组合方案三B，日频上相乘，月度取均值，信息比率1.92'''###
prod_dailyB=tr*pct
prod_daily_to_monthlyB=prod_dailyB.rolling(20).mean().resample('M').last()
prod_daily_to_monthlyB_neu=fall.get_neutral_monthly_factors(prod_daily_to_monthlyB,boxcox=True)
shen_prod_daily_to_monthlyB=pure_moonnight(prod_daily_to_monthlyB_neu,10,filename='日_原始换手乘原始影差_mean')

'''组合方案三C，日频上标准化归正相乘，月度取标准差，信息比率2.37'''###
prod_daily_to_monthly_standardA_std=prod_daily_standardA.rolling(20).std().resample('M').last()
prod_daily_to_monthly_standardA_std_neu=fall.get_neutral_monthly_factors(prod_daily_to_monthly_standardA_std,boxcox=True)
shen_prod_daily_standardA_std=pure_moonnight(prod_daily_to_monthly_standardA_std_neu,10,filename='日标_原始换手乘原始影差_std')

'''组合方案三D，日频上相乘，月度取标准差，信息比率1.91'''###
prod_daily_to_monthlyB_std=prod_dailyB.rolling(20).std().resample('M').last()
prod_daily_to_monthlyB_std_neu=fall.get_neutral_monthly_factors(prod_daily_to_monthlyB_std,boxcox=True)
shen_prod_daily_standardB_std=pure_moonnight(prod_daily_to_monthlyB_std_neu,10,filename='日_原始换手乘原始影差_std')

'''组合方案三E，B与C的结合，月度标准化相加，信息比率2.89'''###
fall_third_plus_monthlyBC=pure_fallmount(prod_daily_to_monthlyB_neu)+(pure_fallmount(prod_daily_to_monthly_standardA_std_neu),)
shen_third_plus_monthlyBC=pure_moonnight(fall_third_plus_monthlyBC(),10,filename='月_mean加标std')

'''组合方案三EE，B与D的结合，月度标准化相加，信息比率2.58'''
fall_third_plus_monthlyBD=pure_fallmount(prod_daily_to_monthlyB_neu)+(pure_fallmount(prod_daily_to_monthlyB_std_neu),)
shen_third_plus_monthlyBD=pure_moonnight(fall_third_plus_monthlyBD(),10,filename='月_mean加std')

'''组合方案三F，B与C的结合，月度标准化归正相乘，信息比率1.92'''
fall_third_prod_monthlyBC=pure_fallmount(prod_daily_to_monthlyB_neu)*(pure_fallmount(prod_daily_to_monthly_standardA_std_neu),)
shen_third_prod_monthlyBC=pure_moonnight(fall_third_prod_monthlyBC(),10)

'''组合方案三G，B与C的结合，月度相乘，信息比率0.3'''
fall_third_prod_direct_monthlyBC=prod_daily_to_monthlyB_neu*prod_daily_to_monthly_standardA_std_neu
shen_third_prod_direct_monthlyBC=pure_moonnight(fall_third_prod_direct_monthlyBC,10)

'''组合方案一与组合方案三E的结合，月度标准化相加，信息比率2.64'''
fall_first_third_plus=fall_plus_monthly+(fall_third_plus_monthlyBC,)
shen_first_third_plus=pure_moonnight(fall_first_third_plus(),10)

'''组合方案一与组合方案三E的结合，月度标准化归正相乘，信息比率2.51'''
fall_first_third_prod=fall_plus_monthly*(fall_third_plus_monthlyBC,)
shen_first_third_prod=pure_moonnight(fall_first_third_prod(),10)

'''组合方案一与组合方案三E的结合，月度相乘，信息比率0.09'''
fall_first_third_prod_direct=fall_plus_monthly()*fall_third_plus_monthlyBC()
shen_first_third_prod_direct=pure_moonnight(fall_first_third_prod_direct,10)

'''组合方案三H，B与C的结合，月度排序值相加，信息比率2.66'''
third_plus_monthlyBC_rank=prod_daily_to_monthlyB_neu.rank(axis=1)+prod_daily_to_monthly_standardA_std_neu.rank(axis=1)
shen_third_plus_monthlyBC_rank=pure_moonnight(third_plus_monthlyBC_rank,10)

'''组合方案一rank与组合方案三Hrank的结合，排序值相加，信息比率2.24'''
first_rank_third_rank_plus=fall_plus_monthly_rank+third_plus_monthlyBC_rank
shen_first_rank_third_rank_plus=pure_moonnight(first_rank_third_rank_plus,10)

'''组合方案三I，B与C的结合，月度排序值相乘，信息比率2.46'''
third_prod_monthlyBC_rank=prod_daily_to_monthlyB_neu.rank(axis=1)*prod_daily_to_monthly_standardA_std_neu.rank(axis=1)
shen_third_prod_monthlyBC_rank=pure_moonnight(third_prod_monthlyBC_rank,10)

'''组合方案一rank与组合方案三Irank的结合，排序值相乘，信息比率1.93'''
first_rank_third_rank_prod=fall_plus_monthly_rank*third_prod_monthlyBC_rank
shen_first_rank_third_rank_prod=pure_moonnight(first_rank_third_rank_prod,10)

'''组合方案一rank与组合方案三Irank的结合，排序值标准化归正相乘，信息比率2.35'''
first_rank_third_rank_prod_equal=pure_fallmount(fall_plus_monthly_rank)*(pure_fallmount(third_prod_monthlyBC_rank),)
shen_first_rank_third_rank_prod_equal=pure_moonnight(first_rank_third_rank_prod_equal(),10)

'''组合方案五，20天的因子序列，求相关系数，信息比率0.04'''###
class dawn_tr_pct_corr(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_corr,self).__init__(fac1=fac1,fac2=fac2)

    def corr_me(self,df):
        '''计算相关系数，生成月度因子'''
        corr=df.corr().iloc[0,1]
        return corr

tr_pct_corr=dawn_tr_pct_corr(tr,pct)
tr_pct_corr.run(tr_pct_corr.corr_me)

shen_tr_pct_corr=pure_moonnight(tr_pct_corr(),10,boxcox=True,filename='原始换手corr原始影差')


'''组合方案四，因子切割，正向的pct与负向的pct，正向信息比率1.62，负向信息比率1.63，组合后1.63'''###
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def positive_cut(self,df):
        '''pct为正的部分，生成月度因子'''
        df=df[df.fac2>0]
        return df.fac1.mean()

    def negative_cut(self,df):
        '''pct为负的部分，生成月度因子'''
        df=df[df.fac2<0]
        return df.fac1.mean()

tr_pct_cut_positive=dawn_tr_pct_cut(tr,pct)
tr_pct_cut_positive.run(tr_pct_cut_positive.positive_cut)
tr_pct_cut_negative=dawn_tr_pct_cut(tr,pct)
tr_pct_cut_negative.run(tr_pct_cut_negative.negative_cut)

shen_tr_pct_cut_positive=pure_moonnight(tr_pct_cut_positive(),10,boxcox=True,filename='正切割原始换手')
shen_tr_pct_cut_negative=pure_moonnight(tr_pct_cut_negative(),10,boxcox=True,filename='负切割原始换手')
fall_tr_pct_cut_pos_neg=pure_fallmount(tr_pct_cut_negative())+(pure_fallmount(tr_pct_cut_positive()),)
shen_tr_pct_cut_pos_neg=pure_moonnight(fall_tr_pct_cut_pos_neg(),10,boxcox=True,filename='正负切割原始换手')

'''组合方案四，因子切割，正向的pct与负向的pct，用天数加权，即乘以天数，正向信息比率1.89，负向信息比率1.41，组合后信息比率1.62'''###
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def positive_cut(self,df):
        '''pct为正的部分，生成月度因子'''
        df=df[df.fac2>0]
        return df.fac1.mean()*df.shape[0]

    def negative_cut(self,df):
        '''pct为负的部分，生成月度因子'''
        df=df[df.fac2<0]
        return df.fac1.mean()*df.shape[0]

tr_pct_cut_positive_weighted=dawn_tr_pct_cut(tr,pct)
tr_pct_cut_positive_weighted.run(tr_pct_cut_positive_weighted.positive_cut)
tr_pct_cut_negative_weighted=dawn_tr_pct_cut(tr,pct)
tr_pct_cut_negative_weighted.run(tr_pct_cut_negative_weighted.negative_cut)

shen_tr_pct_cut_positive_weighted=pure_moonnight(tr_pct_cut_positive_weighted(),10,boxcox=True,filename='加权正切割原始换手')
shen_tr_pct_cut_negative_weighted=pure_moonnight(tr_pct_cut_negative_weighted(),10,boxcox=True,filename='加权负切割原始换手')
fall_tr_pct_cut_pos_neg_weighted=pure_fallmount(tr_pct_cut_positive_weighted())+(pure_fallmount(tr_pct_cut_negative_weighted()),)
shen_tr_pct_cut_pos_neg_weighted=pure_moonnight(fall_tr_pct_cut_pos_neg_weighted(),10,boxcox=True,filename='加权正负切割原始换手')

'''组合方案四，因子切割，最大的四个pct与最小的4个pct，最大的信息比率1.85，最小的信息比率1.89，标准化相加信息比率1.90'''###还没跑
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(4,'fac2')
        return df.fac1.mean()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(4,'fac2')
        return df.fac1.mean()

tr_pct_cut_max=dawn_tr_pct_cut(tr,pct)
tr_pct_cut_max.run(tr_pct_cut_max.max_cut)
tr_pct_cut_min=dawn_tr_pct_cut(tr,pct)
tr_pct_cut_min.run(tr_pct_cut_min.min_cut)

shen_tr_pct_cut_max=pure_moonnight(tr_pct_cut_max(),10,boxcox=True,filename='最大切割原始换手')
shen_tr_pct_cut_min=pure_moonnight(tr_pct_cut_min(),10,boxcox=True,filename='最小切割原始换手')
fall_tr_pct_cut_max_min=pure_fallmount(tr_pct_cut_max())+(pure_fallmount(tr_pct_cut_min()),)
shen_tr_pct_cut_max_min=pure_moonnight(fall_tr_pct_cut_max_min(),10,boxcox=True,filename='大小切割原始换手')

'''方案一A、方案二A结合，标准化相加，信息比率2.30'''
fall_first_second_plus=fall_plus_monthly+(fall_prod_monthlyA,)
shen_first_second_plus=pure_moonnight(fall_first_second_plus(),10)

'''方案一A、方案三B、方案三C结合，标准化相加，信息比率'''
fall_first_third_plus_ABC_equal=fall_plus_monthly+(pure_fallmount(prod_daily_to_monthlyB_neu),pure_fallmount(prod_daily_to_monthly_standardA_std_neu))
shen_first_third_plus_ABC_equal=pure_moonnight(fall_first_third_plus_ABC_equal(),10)

'''方案一A和方案一B结合，标准化相加，信息比率2.58'''
fall_first_plus_AB=fall_plus_monthly+(pure_fallmount(fall_plus_monthly_rank),)
shen_first_plus_AB=pure_moonnight(fall_first_plus_AB(),10)

'''方案一A、方案一B、方案三B、方案三C结合，标准化相加，信息比率2.57'''
fall_first_third_plus_ABC_equal=fall_plus_monthly+(pure_fallmount(prod_daily_to_monthlyB),pure_fallmount(prod_daily_to_monthly_standardA_std),pure_fallmount(fall_plus_monthly_rank))
shen_first_third_plus_ABBC_equal=pure_moonnight(fall_first_third_plus_ABC_equal(),10)

'''方案一A和方案一B组合，方案三B和方案三C结合，然后二者结合，信息比率2.58'''
fall_first_AB_third_BC_two_step=(fall_plus_monthly+(pure_fallmount(fall_plus_monthly_rank),))+(pure_fallmount(prod_daily_to_monthlyB)+(pure_fallmount(prod_daily_to_monthly_standardA_std),),)
shen_first_AB_third_BC_two_step=pure_moonnight(fall_first_AB_third_BC_two_step(),10)

'''方案一A排序值与方案一B排序值，相加，得到新因子，信息比率2.44'''
fall_first_plus_AB_rank=fall_plus_monthly().rank(axis=1)+fall_plus_monthly_rank.rank(axis=1)
shen_first_plus_AB_rank=pure_moonnight(fall_first_plus_AB_rank,10)

'''方案一A与方案一B的标准化相加，得到新因子1，方案一A与方案一B的排序值相加，得到新因子2，将1和2标准化相加，信息比率2.55'''
fall_first_plus_AB_rank_repeat=fall_first_plus_AB+(pure_fallmount(fall_first_plus_AB_rank),)
shen_first_plus_AB_rank_repeat=pure_moonnight(fall_first_plus_AB_rank_repeat(),10)














'''用tr和pct互相回归取参数，均剔除对方对自己的影响，得到纯净换手率etr和纯净上下影线差epct'''###
tr_long=tr.stack().reset_index()
tr_long.columns=['date','code','tr']
pct_long=pct.stack().reset_index()
pct_long.columns=['date','code','pct']
tr_pct_long=pd.merge(tr_long,pct_long,on=['date','code'])

tqdm.tqdm.pandas()
corr_here=tr_pct_long.groupby('date').progress_apply(lambda x:x.corr().iloc[0,1])
corr_here.mean()

def get_e(df):
    y_tr=smf.ols('tr~pct',data=df).fit()
    a_tr=y_tr.params['Intercept']
    b_tr=y_tr.params['pct']
    df=df.assign(etr=df.tr-df.pct*b_tr-a_tr)
    y_pct=smf.ols('pct~tr',data=df).fit()
    a_pct=y_pct.params['Intercept']
    b_pct=y_pct.params['tr']
    df=df.assign(epct=df.pct-df.tr*b_pct-a_pct)
    return df

tqdm.tqdm.pandas()
tr_pct_long=tr_pct_long.groupby('date').progress_apply(get_e)
etr=tr_pct_long[['date','code','etr']]
epct=tr_pct_long[['date','code','epct']]
etr=etr.set_index(['date','code']).unstack()
etr.columns=[i[1] for i in list(etr.columns)]
epct=epct.set_index(['date','code']).unstack()
epct.columns=[i[1] for i in list(epct.columns)]


'''纯净换手率信息比率1.68，纯净上下影线差信息比率2.34，换手率的标准差2.51，上下影线差标准差1.96'''###
etr_monthly=etr.rolling(20).std().resample('M').last()
epct_monthly=epct.rolling(20).mean().resample('M').last()
fall=pure_fall()
etr_monthly_neu=fall.get_neutral_monthly_factors(etr_monthly,boxcox=True)
epct_monthly_neu=fall.get_neutral_monthly_factors(epct_monthly,boxcox=True)
shen_etr_monthly=pure_moonnight(etr_monthly_neu,10,filename='纯净换手率')
shen_epct_monthly=pure_moonnight(epct_monthly_neu,10,filename='纯净上下影线差')

'''纯净换手率和纯净上下影线差，标准化相加，信息比率2.25'''###
fall_etr_epct_monthly_plus=pure_fallmount(etr_monthly_neu)+(pure_fallmount(epct_monthly_neu),)
shen_etr_epct_monthly_plus=pure_moonnight(fall_etr_epct_monthly_plus(),10,filename='月标_纯净换手加纯净影差')

'''纯净换手率和纯净上下影线差，标准化归正相乘，信息比率2.5'''###
fall_etr_epct_monthly_prod=pure_fallmount(etr_monthly_neu)*(pure_fallmount(epct_monthly_neu),)
shen_etr_epct_monthly_prod=pure_moonnight(fall_etr_epct_monthly_prod(),10,filename='月标_纯净换手乘纯净影差')

'''纯净换手率和纯净上下影线差，排序值相加，信息比率2.26'''
fall_etr_epct_monthly_plus_rank=etr_monthly_neu.rank(axis=1)+epct_monthly_neu.rank(axis=1)
shen_etr_epct_monthly_plus_rank=pure_moonnight(fall_etr_epct_monthly_plus_rank,10)

'''纯净换手率和纯净上下影线差，排序值相乘，信息比率2.24'''
fall_etr_epct_monthly_plus_rank=etr_monthly_neu.rank(axis=1)*epct_monthly_neu.rank(axis=1)
shen_etr_epct_monthly_plus_rank=pure_moonnight(fall_etr_epct_monthly_plus_rank,10)

'''百分组表格'''###
home=pure_newyear(etr_monthly_neu,epct_monthly_neu,10)
alongx=home().mean(axis=1)
alongy=home().mean()

'''组合方案三A，日频上标准化归正相乘，月度取均值，信息比率2.05'''###
etr_daily_standard=fall.standardlize_in_cross_section(etr)
etr_daily_standard=(etr_daily_standard.T-etr_daily_standard.T.min()).T
epct_daily_standard=fall.standardlize_in_cross_section(epct)
epct_daily_standard=(epct_daily_standard.T-etr_daily_standard.T.min()).T
eprod_daily_standardA=etr_daily_standard*epct_daily_standard
eprod_daily_to_monthly_standardA=eprod_daily_standardA.rolling(20).mean().resample('M').last()
eprod_daily_to_monthly_standardA_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthly_standardA,boxcox=True)
shen_eprod_daily_standardA=pure_moonnight(eprod_daily_to_monthly_standardA_neu,10,filename='日标_纯净换手乘纯净影差_mean')

'''组合方案三B，日频上相乘，月度取均值，信息比率0.27'''###
eprod_dailyB=etr*epct
eprod_daily_to_monthlyB=eprod_dailyB.rolling(20).mean().resample('M').last()
eprod_daily_to_monthlyB_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthlyB,boxcox=True)
shen_eprod_daily_to_monthlyB=pure_moonnight(eprod_daily_to_monthlyB_neu,10,filename='日_纯净换手乘纯净影差_mean')

'''组合方案三C，日频上标准化归正相乘，月度取标准差，信息比率2.52'''###
eprod_daily_to_monthly_standardA_std=eprod_daily_standardA.rolling(20).std().resample('M').last()
eprod_daily_to_monthly_standardA_std_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthly_standardA_std,boxcox=True)
shen_eprod_daily_standardA_std=pure_moonnight(eprod_daily_to_monthly_standardA_std_neu,10,filename='日标_纯净换手乘纯净影差_std')

'''组合方案三D，日频上相乘，月度取标准差，信息比率2.35'''###
eprod_daily_to_monthlyB_std=eprod_dailyB.rolling(20).std().resample('M').last()
eprod_daily_to_monthlyB_std_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthlyB_std,boxcox=True)
shen_eprod_daily_standardB_std=pure_moonnight(eprod_daily_to_monthlyB_std_neu,10,filename='日_纯净换手乘纯净影差_std')

'''组合方案三E，A与C的结合，月度标准化相加，信息比率2.35'''###
fall_ethird_plus_monthlyAC=pure_fallmount(eprod_daily_to_monthly_standardA_neu)+(pure_fallmount(eprod_daily_to_monthly_standardA_std_neu),)
shen_ethird_plus_monthlyAC=pure_moonnight(fall_ethird_plus_monthlyAC(),10,filename='月标_mean加std')

'''组合方案三F，A与C的结合，月度标准化相乘，信息比率2.74'''###
fall_ethird_prod_monthlyAC=pure_fallmount(eprod_daily_to_monthly_standardA_neu)*(pure_fallmount(eprod_daily_to_monthly_standardA_std_neu),)
shen_ethird_prod_monthlyAC=pure_moonnight(fall_ethird_prod_monthlyAC(),10,filename='月标_mean乘std')


'''组合方案四，因子切割，最大的四个pct与最小的4个pct，最大的信息比率2.06，最小的信息比率2.23'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(4,'fac2')
        return df.fac1.mean()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(4,'fac2')
        return df.fac1.mean()

etr_epct_cut_max=dawn_tr_pct_cut(epct,etr)
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(epct,etr)
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')


'''组合方案四，因子切割，最大的四个pct与最小的10个pct，最大的信息比率2.12，最小的信息比率2.25，'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(10,'fac2')
        return df.fac1.mean()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(10,'fac2')
        return df.fac1.mean()

etr_epct_cut_max=dawn_tr_pct_cut(epct,etr.shift(1))
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(epct,etr.shift(1))
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')

fall_epct_plus=etr_epct_cut_max().rank(axis=1)+etr_epct_cut_min().rank(axis=1)
shen_epct_plus=pure_moonnight(fall_epct_plus,10)

'''组合方案四，因子切割，最大的四个pct与最小的4个pct，最大的信息比率1.97，最小的信息比率1.87，标准化相加信息比率1.93，直接相加1.94，标准化归正相乘2.50'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(4,'fac2')
        return df.fac1.mean()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(4,'fac2')
        return df.fac1.mean()

etr_epct_cut_max=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')
prod_etr_epct_cut_max_min=pure_fallmount(etr_epct_cut_min())*(pure_fallmount(etr_epct_cut_max()),)
fall_etr_epct_cut_max_min=pure_fallmount(etr_epct_cut_max())+(pure_fallmount(etr_epct_cut_min()),)
shen_etr_epct_cut_max_min=pure_moonnight(prod_etr_epct_cut_max_min(),10,boxcox=True)



'''组合方案四，因子切割，正向的pct与负向的pct，用天数加权，即乘以天数，正向信息比率2.07，负向信息比率，组合后信息比率'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def positive_cut(self,df):
        '''pct为正的部分，生成月度因子'''
        df=df[df.fac2>0]
        return df.fac1.mean()*df.shape[0]

    def negative_cut(self,df):
        '''pct为负的部分，生成月度因子'''
        df=df[df.fac2<0]
        return df.fac1.mean()*df.shape[0]

etr_epct_cut_positive_weighted=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_positive_weighted.run(etr_epct_cut_positive_weighted.positive_cut)
etr_epct_cut_negative_weighted=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_negative_weighted.run(etr_epct_cut_negative_weighted.negative_cut)

shen_etr_epct_cut_positive_weighted=pure_moonnight(etr_epct_cut_positive_weighted(),10,boxcox=True,filename='加权正切割原始换手')
shen_etr_epct_cut_negative_weighted=pure_moonnight(etr_epct_cut_negative_weighted(),10,boxcox=True,filename='加权负切割原始换手')
fall_etr_epct_cut_pos_neg_weighted=pure_fallmount(etr_epct_cut_positive_weighted())+(pure_fallmount(etr_epct_cut_negative_weighted()),)
shen_etr_epct_cut_pos_neg_weighted=pure_moonnight(fall_etr_epct_cut_pos_neg_weighted(),10,boxcox=True,filename='加权正负切割原始换手')

















'''纯净换手率信息比率1.89，纯净上下影线差信息比率2.34，换手率的标准差2.77，上下影线差标准差1.96'''###
etr_monthly_std=etr.rolling(20).std().resample('M').last()
etr_monthly_mean=etr.rolling(20).mean().resample('M').last()
epct_monthly_mean=epct.rolling(20).mean().resample('M').last()
epct_monthly_std=epct.rolling(20).std().resample('M').last()
fall=pure_fall()
etr_monthly_mean_neu=fall.get_neutral_monthly_factors(etr_monthly_mean,boxcox=True)
etr_monthly_std_neu=fall.get_neutral_monthly_factors(etr_monthly_std,boxcox=True)
epct_monthly_mean_neu=fall.get_neutral_monthly_factors(epct_monthly_mean,boxcox=True)
epct_monthly_std_neu=fall.get_neutral_monthly_factors(epct_monthly_std,boxcox=True)
shen_etr_monthly=pure_moonnight(etr_monthly_neu,10,filename='纯净换手率')
shen_epct_monthly=pure_moonnight(epct_monthly_neu,10,filename='纯净上下影线差')

'''纯净换手率和纯净上下影线差，标准化相加，信息比率2.83'''###
fall_etr_epct_monthly_plus=pure_fallmount(etr_monthly_std_neu)+(pure_fallmount(epct_monthly_mean_neu),)
shen_etr_epct_monthly_plus=pure_moonnight(fall_etr_epct_monthly_plus(),10,filename='月标_纯净换手加纯净影差')

'''纯净换手率和纯净上下影线差，标准化归正相乘，信息比率3.03'''###
fall_etr_epct_monthly_prod=pure_fallmount(etr_monthly_mean_neu)*(pure_fallmount(etr_monthly_std_neu),pure_fallmount(epct_monthly_std_neu),pure_fallmount(epct_monthly_mean_neu))
shen_etr_epct_monthly_prod=pure_moonnight(fall_etr_epct_monthly_prod(),10,filename='月标_纯净换手乘纯净影差')

'''纯净换手率和纯净上下影线差，标准化归正相乘，信息比率3.15'''###
fall_etr_epct_monthly_prod=pure_fallmount(etr_monthly_std_neu)*(pure_fallmount(epct_monthly_mean_neu),)
shen_etr_epct_monthly_prod=pure_moonnight(fall_etr_epct_monthly_prod(),10,filename='月标_纯净换手乘纯净影差')

'''纯净换手率和纯净上下影线差，排序值相加，信息比率2.67'''
fall_etr_epct_monthly_plus_rank=etr_monthly_neu.rank(axis=1)+epct_monthly_neu.rank(axis=1)
shen_etr_epct_monthly_plus_rank=pure_moonnight(fall_etr_epct_monthly_plus_rank,10)

'''纯净换手率和纯净上下影线差，排序值相乘，信息比率2.24'''
fall_etr_epct_monthly_plus_rank=etr_monthly_neu.rank(axis=1)*epct_monthly_neu.rank(axis=1)
shen_etr_epct_monthly_plus_rank=pure_moonnight(fall_etr_epct_monthly_plus_rank,10)

'''百分组表格'''###
home=pure_newyear(etr_monthly_neu,epct_monthly_neu,10)
alongx=home().mean(axis=1)
alongy=home().mean()

'''组合方案三A，日频上标准化归正相乘，月度取均值，信息比率2.05'''###
etr_daily_standard=fall.standardlize_in_cross_section(etr)
etr_daily_standard=(etr_daily_standard.T-etr_daily_standard.T.min()).T
epct_daily_standard=fall.standardlize_in_cross_section(epct)
epct_daily_standard=(epct_daily_standard.T-etr_daily_standard.T.min()).T
eprod_daily_standardA=etr_daily_standard*epct_daily_standard
eprod_daily_to_monthly_standardA=eprod_daily_standardA.rolling(20).mean().resample('M').last()
eprod_daily_to_monthly_standardA_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthly_standardA,boxcox=True)
shen_eprod_daily_standardA=pure_moonnight(eprod_daily_to_monthly_standardA_neu,10,filename='日标_纯净换手乘纯净影差_mean')

'''组合方案三B，日频上相乘，月度取均值，信息比率0.27'''###
eprod_dailyB=etr*epct
eprod_daily_to_monthlyB=eprod_dailyB.rolling(20).mean().resample('M').last()
eprod_daily_to_monthlyB_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthlyB,boxcox=True)
shen_eprod_daily_to_monthlyB=pure_moonnight(eprod_daily_to_monthlyB_neu,10,filename='日_纯净换手乘纯净影差_mean')

'''组合方案三C，日频上标准化归正相乘，月度取标准差，信息比率2.52'''###
eprod_daily_to_monthly_standardA_std=eprod_daily_standardA.rolling(20).std().resample('M').last()
eprod_daily_to_monthly_standardA_std_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthly_standardA_std,boxcox=True)
shen_eprod_daily_standardA_std=pure_moonnight(eprod_daily_to_monthly_standardA_std_neu,10,filename='日标_纯净换手乘纯净影差_std')

'''组合方案三D，日频上相乘，月度取标准差，信息比率2.35'''###
eprod_daily_to_monthlyB_std=eprod_dailyB.rolling(20).std().resample('M').last()
eprod_daily_to_monthlyB_std_neu=fall.get_neutral_monthly_factors(eprod_daily_to_monthlyB_std,boxcox=True)
shen_eprod_daily_standardB_std=pure_moonnight(eprod_daily_to_monthlyB_std_neu,10,filename='日_纯净换手乘纯净影差_std')

'''组合方案三E，A与C的结合，月度标准化相加，信息比率2.35'''###
fall_ethird_plus_monthlyAC=pure_fallmount(eprod_daily_to_monthly_standardA_neu)+(pure_fallmount(eprod_daily_to_monthly_standardA_std_neu),)
shen_ethird_plus_monthlyAC=pure_moonnight(fall_ethird_plus_monthlyAC(),10,filename='月标_mean加std')

'''组合方案三F，A与C的结合，月度标准化相乘，信息比率2.74'''###
fall_ethird_prod_monthlyAC=pure_fallmount(eprod_daily_to_monthly_standardA_neu)*(pure_fallmount(eprod_daily_to_monthly_standardA_std_neu),)
shen_ethird_prod_monthlyAC=pure_moonnight(fall_ethird_prod_monthlyAC(),10,filename='月标_mean乘std')


'''组合方案四，因子切割，最大的四个pct与最小的4个pct，最大的信息比率2.06，最小的信息比率2.23'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(4,'fac2')
        return df.fac1.mean()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(4,'fac2')
        return df.fac1.mean()

etr_epct_cut_max=dawn_tr_pct_cut(epct,etr)
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(epct,etr)
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')


'''组合方案四，因子切割，最大的四个pct与最小的10个pct，最大的信息比率2.12，最小的信息比率2.25，'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(10,'fac2')
        return df.fac1.mean()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(10,'fac2')
        return df.fac1.mean()

etr_epct_cut_max=dawn_tr_pct_cut(epct,etr.shift(1))
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(epct,etr.shift(1))
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')

fall_epct_plus=etr_epct_cut_max().rank(axis=1)+etr_epct_cut_min().rank(axis=1)
shen_epct_plus=pure_moonnight(fall_epct_plus,10)

'''组合方案四，因子切割，最大的四个pct与最小的4个pct，最大的信息比率1.97，最小的信息比率1.87，标准化相加信息比率1.93，直接相加1.94，标准化归正相乘2.50'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(4,'fac2')
        return df.fac1.mean()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(4,'fac2')
        return df.fac1.mean()

etr_epct_cut_max=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')
prod_etr_epct_cut_max_min=pure_fallmount(etr_epct_cut_min())*(pure_fallmount(etr_epct_cut_max()),)
fall_etr_epct_cut_max_min=pure_fallmount(etr_epct_cut_max())+(pure_fallmount(etr_epct_cut_min()),)
shen_etr_epct_cut_max_min=pure_moonnight(prod_etr_epct_cut_max_min(),10,boxcox=True)



'''组合方案四，因子切割，正向的pct与负向的pct，用天数加权，即乘以天数，正向信息比率2.07，负向信息比率，组合后信息比率'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def positive_cut(self,df):
        '''pct为正的部分，生成月度因子'''
        df=df[df.fac2>0]
        return df.fac1.mean()*df.shape[0]

    def negative_cut(self,df):
        '''pct为负的部分，生成月度因子'''
        df=df[df.fac2<0]
        return df.fac1.mean()*df.shape[0]

etr_epct_cut_positive_weighted=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_positive_weighted.run(etr_epct_cut_positive_weighted.positive_cut)
etr_epct_cut_negative_weighted=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_negative_weighted.run(etr_epct_cut_negative_weighted.negative_cut)

shen_etr_epct_cut_positive_weighted=pure_moonnight(etr_epct_cut_positive_weighted(),10,boxcox=True,filename='加权正切割原始换手')
shen_etr_epct_cut_negative_weighted=pure_moonnight(etr_epct_cut_negative_weighted(),10,boxcox=True,filename='加权负切割原始换手')
fall_etr_epct_cut_pos_neg_weighted=pure_fallmount(etr_epct_cut_positive_weighted())+(pure_fallmount(etr_epct_cut_negative_weighted()),)
shen_etr_epct_cut_pos_neg_weighted=pure_moonnight(fall_etr_epct_cut_pos_neg_weighted(),10,boxcox=True,filename='加权正负切割原始换手')








'''组合方案四，因子切割，最大的四个pct与最小的4个pct，最大的信息比率2.83，最小的信息比率2.22，标准化相加信息比率，直接相加，标准化归正相乘'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(4,'fac2')
        return df.fac1.std()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(4,'fac2')
        return df.fac1.std()

etr_epct_cut_max=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(etr,epct)
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')
prod_etr_epct_cut_max_min=pure_fallmount(etr_epct_cut_min())*(pure_fallmount(etr_epct_cut_max()),)
fall_etr_epct_cut_max_min=pure_fallmount(etr_epct_cut_max())+(pure_fallmount(etr_epct_cut_min()),)
shen_etr_epct_cut_max_min=pure_moonnight(fall_etr_epct_cut_max_min(),10,boxcox=True)
shen_etr_epct_cut_max_min=pure_moonnight(prod_etr_epct_cut_max_min(),10,boxcox=True)




'''组合方案四，因子切割，最大的四个pct与最小的10个pct，最大的信息比率2.06，最小的信息比率1.55，'''
class dawn_tr_pct_cut(pure_dawn):
    def __init__(self,fac1,fac2):
        self.fac1=fac1
        self.fac2=fac2
        super(dawn_tr_pct_cut,self).__init__(fac1=fac1,fac2=fac2)

    def max_cut(self,df):
        '''上下影线差距最大的部分对应的换手率'''
        df=df.nlargest(10,'fac2')
        return df.fac1.std()

    def min_cut(self,df):
        '''上下影线差距最小的部分对应的换手率'''
        df=df.nsmallest(10,'fac2')
        return df.fac1.std()

etr_epct_cut_max=dawn_tr_pct_cut(epct,etr)
etr_epct_cut_max.run(etr_epct_cut_max.max_cut)
etr_epct_cut_min=dawn_tr_pct_cut(epct,etr)
etr_epct_cut_min.run(etr_epct_cut_min.min_cut)

shen_etr_epct_cut_max=pure_moonnight(etr_epct_cut_max(),10,boxcox=True,filename='最大切割纯净换手')
shen_etr_epct_cut_min=pure_moonnight(etr_epct_cut_min(),10,boxcox=True,filename='最小切割纯净换手')



###纯净切割反过来
###方案一直接相加时，用epct加tr
###等权正负切割纯净
###加名字和写入excel
###切割时求标准差
###基准因子求标准差
###对纯净换手做切割
