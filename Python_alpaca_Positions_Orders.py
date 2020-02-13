import os
import alpaca_trade_api as tradeapi
import pandas as pd
import sys
from PySide2.QtGui import *
from PySide2.QtCore import *
from PySide2.QtWidgets import *

def loadOpenOrders():
    open_orders = api.list_orders(status='open', limit = 500, direction = 'desc')
    return open_orders

def loadOpenPosition():
    existing_positions = api.list_positions()
    return existing_positions

def loadClosedOrders():
    closed_orders= api.list_orders(status = 'closed', limit = 500, direction = 'desc')
    return closed_orders

class Tree(QWidget):

    def __init__(self,columns):
        QWidget.__init__(self)

        self.tv = QTreeView()
        self.tv.setSortingEnabled(True)
        self.tv.setAlternatingRowColors(True)
        self.tvim = QStandardItemModel()
        self.tvim.setHorizontalHeaderLabels(columns)
        self.tv.setModel(self.tvim)
        MainWindow = QHBoxLayout(self)
        MainWindow.addWidget(self.tv)
        self.setLayout(MainWindow)

    def addrow(self,columns):
        items = []
        for c in columns:
            items.append(QStandardItem(c))
        self.tvim.appendRow(tuple(items))
        return items

    def addchild(self,parent,columns):
        items = []
        for c in columns:
            items.append(QStandardItem(c))
        parent.appendRow(tuple(items))
        return items

if __name__ == "__main__":

    try:
        key_id =  os.environ['KEY_ID']
        secret_key = os.environ['SECRET_KEY']
        base_url = os.environ['BASE_URL']
    except Exception as e:
        raise Exception('Set API keys')

    # alpaca api
    api = tradeapi.REST(key_id, secret_key, base_url)

    #Open order
    open_orders = loadOpenOrders()
    oord_df = pd.DataFrame([order.__dict__['_raw'] for order in open_orders])

    #closed orders
    closed_orders = loadClosedOrders()
    cord_df=pd.DataFrame([order.__dict__['_raw'] for order in closed_orders])
    cord_df['filled_on']=cord_df['filled_at'].str[:10]

    #last orders
    lord_df=cord_df[['symbol', 'filled_on']]
    lord_df.set_index('symbol',inplace=True)
    lord_s = lord_df.groupby(['symbol'])['filled_on'].first()

    #open position
    open_positions = loadOpenPosition()
    opos_df = pd.DataFrame([pos.__dict__['_raw'] for pos in open_positions])
    opos_df['profit'] = round(opos_df['unrealized_plpc'].astype(float) * 100, 2)
    opos_df=pd.merge(opos_df, lord_s, on='symbol')


    app = QApplication(sys.argv)
    tree = Tree(['symbols','qty','profit','last_filled_on'])
    for row in opos_df.iterrows():
        p  = tree.addrow([row[1]['symbol'],str(row[1]['qty']).rjust(5),str(row[1]['profit']),row[1]['filled_on']])
        cord=cord_df[cord_df['symbol']==row[1]['symbol']]
        for orow in cord.iterrows():
            c = tree.addchild(p[0],[orow[1]['side'],str(orow[1]['filled_qty']).rjust(10),str(orow[1]['filled_avg_price']),orow[1]['filled_at']])
    tree.show()
    # Run the main Qt loop
    sys.exit(app.exec_())

############################################################################
from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import pandas as pd

ts = TimeSeries(key = 'IH4EENERLUFUKJRW', output_format = 'pandas')
data, meta_data = ts.get_intraday(symbol = 'MSFT', interval = '1min', outputsize = 'full')

data.to_csv('stock_data_MSFT_1_5_2019.csv')