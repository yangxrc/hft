import numpy as np
import pandas as pd
from itertools import product
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

class TradingEnv:
    def __init__(self, fixed_volume=200):
        """
        初始化交易环境
        
        Args:
            fixed_volume (int): 固定每笔交易量，默认200股
        """
        self.obs_feature = 54  # 状态空间：4个market特征 + 50个因子
        
        # 动作空间：-1(空头开仓), 0(不交易), 1(多头开仓)
        self.act_list = list(product([-1, 0, 1]))
        self.act_n = len(self.act_list)
        
        # 交易费率
        self.buy_premium = 0.0002
        self.sell_premium = 0.0012
        
        # 固定每笔交易量
        self.fixed_volume = fixed_volume
        
        # 交易限制
        self.netrol = 0  # 当前净持仓
        self.netrol_limit = 1000  # 持仓限制
        self.cash = 0  # 当前现金
        
        # 交易时间窗口（精确到毫秒）
        self.morning_start = 93000000000  # 9:30:00.000
        self.morning_end = 113000000000   # 11:30:00.000
        self.afternoon_start = 130000000000  # 13:00:00.000
        self.afternoon_end = 150000000000    # 15:00:00.000
        
        # 市场数据
        self.market_data = None
        self.factor_data = None
        self.current_time_idx = 0
        
        # 交易记录
        self.trade_records = []
        
    def _is_trading_time(self, timestamp):
        """判断是否在交易时段"""
        return (self.morning_start <= timestamp <= self.morning_end) or \
               (self.afternoon_start <= timestamp <= self.afternoon_end)
               
    def _validate_timestamp_interval(self, current_time, next_time):
        """验证时间戳间隔是否合法"""
        if next_time is None:
            return True
            
        interval = next_time - current_time
        # 允许10ms或20ms的间隔
        return interval in [10000, 20000]  # 毫秒转微秒
        
    def reset(self, market_data, factor_data, initial_cash=1000000):
        """
        重置环境
        
        Args:
            market_data (pd.DataFrame): 市场数据，包含time_stamp, volume, last, turnover等
            factor_data (pd.DataFrame): 因子数据，50个因子
            initial_cash (float): 初始资金
        """
        # 验证时间戳间隔
        timestamps = market_data['time_stamp'].values
        intervals = np.diff(timestamps)
        valid_intervals = np.isin(intervals, [10000, 20000])
        if not np.all(valid_intervals):
            invalid_idx = np.where(~valid_intervals)[0]
            print(f"Warning: Invalid time intervals at indices: {invalid_idx}")
            print(f"Intervals: {intervals[~valid_intervals]}")
        
        self.market_data = market_data
        self.factor_data = factor_data
        self.cash = initial_cash
        self.netrol = 0
        
        # 找到第一个交易时间点
        self.current_time_idx = 0
        while self.current_time_idx < len(market_data):
            if self._is_trading_time(market_data.iloc[self.current_time_idx]['time_stamp']):
                break
            self.current_time_idx += 1
            
        self.trade_records = []
        
        # 获取第一个状态
        initial_state = self._get_state()
        return initial_state
        
    def _get_state(self):
        """获取当前状态"""
        if self.current_time_idx >= len(self.market_data):
            return None
            
        # 市场特征
        market_features = [
            self.market_data.iloc[self.current_time_idx]['volume'],
            self.market_data.iloc[self.current_time_idx]['last'],
            self.market_data.iloc[self.current_time_idx]['turnover'],
            self.netrol / self.netrol_limit  # 当前持仓比例
        ]
        
        # 因子特征
        factor_features = self.factor_data.iloc[self.current_time_idx].values
        
        # 合并特征
        state = np.concatenate([market_features, factor_features])
        return state
        
    def step(self, action):
        """
        执行一步交易
        
        Args:
            action (int): 动作，来自self.act_list
            
        Returns:
            tuple: (下一状态, 奖励, 是否结束, 信息字典)
        """
        if self.current_time_idx >= len(self.market_data):
            return None, 0, True, {}
            
        current_data = self.market_data.iloc[self.current_time_idx]
        current_time = current_data['time_stamp']
        
        # 检查是否在交易时间内
        if not self._is_trading_time(current_time):
            self.current_time_idx += 1
            return self._get_state(), 0, False, {"message": "Not in trading hours"}
            
        # 执行交易
        reward = self._execute_trade(action, current_data)
        
        # 更新时间步并找到下一个有效的交易时间点
        self.current_time_idx += 1
        while self.current_time_idx < len(self.market_data):
            next_time = self.market_data.iloc[self.current_time_idx]['time_stamp']
            if self._is_trading_time(next_time):
                break
            self.current_time_idx += 1
        
        # 获取新状态
        next_state = self._get_state()
        
        # 判断是否结束
        done = (self.current_time_idx >= len(self.market_data)) or \
               (current_time >= self.afternoon_end)
               
        # 如果结束，进行平仓
        if done:
            self._close_positions(current_data)
            
        info = {
            "netrol": self.netrol,
            "cash": self.cash,
            "time": current_time
        }
        
        return next_state, reward, done, info
        
    def _execute_trade(self, action, current_data):
        """执行交易并计算奖励"""
        action_type = self.act_list[action][0]
        reward = 0
        
        if action_type == 0:  # 不交易
            return 0
            
        # 获取当前价格
        current_price = current_data['last']
        
        if action_type == 1:  # 多头开仓
            if self.netrol + self.fixed_volume <= self.netrol_limit:
                trade_price = current_price * (1 + self.buy_premium)
                cost = trade_price * self.fixed_volume
                if cost <= self.cash:
                    self.cash -= cost
                    self.netrol += self.fixed_volume
                    reward = -cost  # 初始奖励为交易成本
                    self._record_trade("buy", trade_price, self.fixed_volume)
                    
        elif action_type == -1:  # 空头开仓
            if abs(self.netrol - self.fixed_volume) <= self.netrol_limit:
                trade_price = current_price * (1 - self.sell_premium)
                revenue = trade_price * self.fixed_volume
                self.cash += revenue
                self.netrol -= self.fixed_volume
                reward = revenue  # 初始奖励为交易收入
                self._record_trade("sell", trade_price, self.fixed_volume)
                
        return reward
        
    def _close_positions(self, current_data):
        """平仓所有持仓"""
        current_price = current_data['last']
        
        if self.netrol > 0:
            trade_price = current_price * (1 - self.sell_premium)
            revenue = trade_price * self.netrol
            self.cash += revenue
            self._record_trade("sell", trade_price, self.netrol)
            self.netrol = 0
        elif self.netrol < 0:
            trade_price = current_price * (1 + self.buy_premium)
            cost = trade_price * abs(self.netrol)
            self.cash -= cost
            self._record_trade("buy", trade_price, abs(self.netrol))
            self.netrol = 0
            
    def _record_trade(self, trade_type, price, volume):
        """记录交易"""
        self.trade_records.append({
            'time': self.market_data.iloc[self.current_time_idx]['time_stamp'],
            'type': trade_type,
            'price': price,
            'volume': volume,
            'netrol': self.netrol,
            'cash': self.cash
        })
        
    def get_trade_records(self):
        """获取交易记录"""
        return pd.DataFrame(self.trade_records)