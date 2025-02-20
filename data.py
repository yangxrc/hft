import os
import numpy as np
import scipy.io as sio
import pandas as pd

class DataProcessor:
    def __init__(self, market_dir, factor_dir, factor_backup_dir):
        """
        初始化数据处理器
        
        Args:
            market_dir (str): market数据的根目录 (/home/public/data_base/step10)
            factor_dir (str): 主factor数据目录 (/home/public/data_share2/StockFactor/375)
            factor_backup_dir (str): 备份factor数据目录 (/home/factorData/stock/375)
        """
        self.market_dir = market_dir
        self.factor_dir = factor_dir
        self.factor_backup_dir = factor_backup_dir
        
    def load_market_data(self, file_path):
        """
        加载market数据
        
        Args:
            file_path (str): mat文件路径
            
        Returns:
            tuple: (时间数组, 特征数组)
        """
        try:
            mat_data = sio.loadmat(file_path)
            time = mat_data['Data']['Time'][0][0].astype(np.float64)
            name_list=mat_data['Data'].dtype.names[:6]
            features = np.column_stack([
                mat_data['Data'][i][0][0].flatten() 
                for i in name_list
            ])
            
            return time, features
        except Exception as e:
            print(f"Error loading market data {file_path}: {str(e)}")
            return None, None
            
    def load_factor_data(self, day, stock):
        """
        加载因子数据，带备份路径
        
        Args:
            day (str): 日期文件夹
            stock (str): 股票代码
            
        Returns:
            tuple: (时间戳数组, 因子数据DataFrame)
        """
        try:
            # 尝试主路径
            factor_path = f'{self.factor_dir}/{day}/{stock}.mat'
            factor_data = sio.loadmat(factor_path)['feature'].astype(np.float64)
        except:
            # 尝试备份路径
            factor_path = f'{self.factor_backup_dir}/{day}/{stock}.mat'
            factor_data = sio.loadmat(factor_path)['feature'].astype(np.float64)
        
        # 提取时间戳（第二列）和因子数据（第三列开始的50个因子）
        factor_time = factor_data[:, 1]
        factor_features = factor_data[:, 2:52]  # 取第3列开始的50个因子
        
        return factor_time, pd.DataFrame(
            factor_features,
            columns=[f'factor_{i}' for i in range(50)]
        )
            
    def process_stock_data(self, stock_code, start_date, end_date):
        """
        处理指定股票在给定时间范围内的数据
        
        Args:
            stock_code (str): 股票代码，如'000001.SZ'
            start_date (str): 开始日期，格式'YYYYMMDD'
            end_date (str): 结束日期，格式'YYYYMMDD'
            
        Returns:
            pd.DataFrame: 合并后的数据框
        """
        # 获取日期范围内的所有文件夹
        date_folders = [d for d in os.listdir(self.market_dir) 
                       if start_date <= d <= end_date]
        
        market_data = []
        factor_data = []
        
        for date_folder in sorted(date_folders):
            # 处理market数据
            market_file = os.path.join(self.market_dir, date_folder, f"{stock_code}.mat")
            if os.path.exists(market_file):
                market_time, market_features = self.load_market_data(market_file)
                if market_time is not None:
                    market_df = pd.DataFrame(market_features, 
                                          columns=[f'market_feature_{i}' for i in range(6)])
                    market_df['timestamp'] = market_time
                    market_data.append(market_df)
            
            # 处理factor数据
            try:
                factor_time, factor_df = self.load_factor_data(date_folder, stock_code)
                factor_df['timestamp'] = factor_time  # 使用factor自己的时间戳
                factor_data.append(factor_df)
            except Exception as e:
                print(f"Error loading factor data for {date_folder}/{stock_code}: {str(e)}")
                continue
        
        # 合并所有数据
        if not market_data or not factor_data:
            print("No data available for the specified period")
            return None
            
        market_df = pd.concat(market_data, ignore_index=True)
        factor_df = pd.concat(factor_data, ignore_index=True)
        
        # 以market数据的时间戳为准进行合并
        merged_df = pd.merge_asof(market_df.sort_values('timestamp'),
                                 factor_df.sort_values('timestamp'),
                                 on='timestamp',
                                 direction='forward')  # 使用前向填充
        
        return merged_df

# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = DataProcessor(
        market_dir="/home/public/data_base/step10",
        factor_dir="/home/public/data_share2/StockFactor/375",
        factor_backup_dir="/home/factorData/stock/375"
    )
    
    # 处理特定股票的数据
    result_df = processor.process_stock_data(
        stock_code="000001.SZ",
        start_date="20240101",
        end_date="20240131"
    )
    
    if result_df is not None:
        print(result_df.head())
        print("\nMarket columns:", [col for col in result_df.columns if 'market' in col])
        print("\nFactor columns:", [col for col in result_df.columns if 'factor' in col])