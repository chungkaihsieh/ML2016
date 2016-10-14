#!/usr/bin/env python

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#first
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

#second
dates = pd.date_range('20130101', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)

df2 = pd.DataFrame({
			'A' : 1.0,
			'B' : pd.Timestamp('20130102'),
			'C' : pd.Series(1,index=list(range(4)),dtype="float32"),
			'D' : np.array([3] * 4, dtype='int32'),
			'E' : pd.Categorical(["test","train","test","train"]),
			'F' : 'foo'
	})

print(df2)
print()
print(df.head())
print(df.tail(3))
print(df.index)
# print(df.columns)
# print(df.describe())

'''selecting by column'''
print(df['A'])
#selecting by rows
print(df[0:3])
print(df['20130102':'20130104'])

'''selection by Label'''
print(df.loc[dates[0]])
print(df.loc[:,['A','B']])
#extract certain component
print(df.loc[dates[0],'A'])
print(df.at[dates[0],'A'])


'''Selection by Position'''
print("selection by Position")
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1,2,4],[0,2]])
print(df.iloc[1:3,:])
print(df.iloc[:,1:3])
print(df.iloc[1,1])

'''Booling Indexing '''
#Using a single column's values to select data
print(df[df.A > 0])
#using isin() method for filtering 
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2)
print(df2[df2['E'].isin( ['two','four'] )])
