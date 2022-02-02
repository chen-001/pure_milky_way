import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from functools import partial
from pure.pure_world import *

class cpv_corr(pure_fall):
    def __init__(self,daily_path='',monthly_path=''):
        '''继承分钟因子的共同特性'''
        super(cpv_corr,self).__init__()
        self.daily_factors_path=self.daily_factors_path+daily_path
        self.monthly_factors_path=self.monthly_factors_path+monthly_path

    def func_cpv_corr(self,df):
        '''计算每一组（一天即为一组）
        在每一组内将分钟数据化为日频因子数据'''
        df=df[['close','amount']]
        return df.corr().iloc[0,1]

    def func_daily_to_monthly_mean(self,df):
        '''将日度因子转化为月度因子
        此处即为每月月底算前20个交易日的相关系数的均值'''
        df=df.rolling(20).mean()
        df=df.resample('M').last()
        return df

    def func_daily_to_monthly_std(self,df):
        '''将日度因子转化为月度因子
        此处即为每月月底算前20个交易日的相关系数的便准查'''
        df=df.rolling(20).apply(np.std)
        df=df.resample('M').last()
        return df

    def trend_in_group(self,l):
        '''对天数次序，做无截距项的回归，取回归系数'''
        y=l.to_numpy()
        x=np.arange(1,21)
        result=sm.OLS(y,x).fit()
        return result.params[0]

    def func_daily_to_monthly_trend(self,df):
        '''保留每一天的趋势系数'''
        tqdm.tqdm.pandas(desc='稍等')
        df=df.rolling(20).progress_apply(self.trend_in_group)
        df=df.resample('M').last()
        return df

# def main(hs300=False,zz500=False):
#市值、ret20、turn20、vol20
bo=pure_moonlight()
to_decap=pure_fallmount(bo('cap_as_factor'))
to_deret20=pure_fallmount(bo('ret20'))
to_deturn20=pure_fallmount(bo('turn20'))
to_devol20=pure_fallmount(bo('vol20'))

#合成均值和标准差因子
cpv_corr_mean=cpv_corr(daily_path='价量相关性.feather',monthly_path='价量相关性均值.feather')
cpv_corr_mean.run(cpv_corr_mean.func_cpv_corr,cpv_corr_mean.func_daily_to_monthly_mean,neutralize=True,hs300=False,zz500=False)
cpv_corr_std=cpv_corr(daily_path='价量相关性.feather',monthly_path='价量相关性标准差.feather')
cpv_corr_std.run(cpv_corr_std.func_cpv_corr,cpv_corr_std.func_daily_to_monthly_std,neutralize=True,hs300=False,zz500=False)
cpv_corr_twins=(cpv_corr_mean+(cpv_corr_std,))()
cpv_corr_here_be=cpv_corr_twins[(cpv_corr_twins.index>=pd.Timestamp('2014-01-01'))&(cpv_corr_twins.index<=pd.Timestamp('2020-01-31'))]

#剔除ret20并回测
cpv_corr_unde_be=pure_fallmount(cpv_corr_here_be)
cpv_corr_de_be=cpv_corr_unde_be-(to_deret20,)
cpv_corr_de_be=cpv_corr_de_be().iloc[:-1,:]
shen_be=pure_moonnight(cpv_corr_de_be,5,plt=False,filename='价量相关性因子5分组')

#计算趋势因子
cpv_corr_trend=cpv_corr(daily_path='价量相关性.feather',monthly_path='价量相关性趋.feather')
cpv_corr_trend.run(cpv_corr_trend.func_cpv_corr,cpv_corr_trend.func_daily_to_monthly_trend,neutralize=False,hs300=False,zz500=False)
# cpv_corr_twins.reset_index().to_feather('D:/因子数据/月频_价量相关性.feather')
cpv_corr_here=cpv_corr_trend()[(cpv_corr_trend().index>=pd.Timestamp('2014-01-01'))&(cpv_corr_trend().index<=pd.Timestamp('2020-01-31'))]

#剔除市值、ret20、turn20、vol20并回测
cpv_corr_unde=pure_fallmount(cpv_corr_here)
cpv_corr_de=cpv_corr_unde-(to_decap,to_deret20,to_deturn20,to_devol20,)
cpv_corr_de=cpv_corr_de().iloc[:-1,:]
shen=pure_moonnight(cpv_corr_de,5,plt=False,filename='价量相关性因子5分组')

#合成均值、标准差因子和趋势因子并回测
deret20=pure_fallmount(cpv_corr_de_be)
trend=pure_fallmount(cpv_corr_de)
final_cpv=deret20+(trend,)
shen_final=pure_moonnight(final_cpv(),5,plt=False,filename='价量相关性因子5分组')

#去除barra因子并回测
snow=pure_snowtrain(final_cpv())
snow_shen=pure_moonnight(snow(),5,plt=False,filename='纯净因子5分组')

snow1=pure_snowtrain(shen_final())
snow_shen1=pure_moonnight(snow(),5,plt=False,filename='纯净因子5分组1')

# main(False,False)
# main(True,False)
# main(False,True)