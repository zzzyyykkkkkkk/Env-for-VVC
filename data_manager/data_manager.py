"""
This module contains the DataManager class for managing and preprocessing
time-series data related to power systems. It includes functionalities for
data loading, cleaning, and basic manipulations.
"""




from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import random
import re


class GeneralPowerDataManager:
    """
    A class to manage and preprocess time series data for power systems.

    Attributes:
        df (pd.DataFrame): The original data.
        data_array (np.ndarray): Array representation of the data.
        active_power_cols (List[str]): List of columns related to active power.
        reactive_power_cols (List[str]): List of columns related to reactive power.
        renewable_active_power_cols (List[str]): List of columns related to renewable active power.
        renewable_reactive_power_cols (List[str]): List of columns related to renewable reactive power.
        price_col (List[str]): List of columns related to price.
        train_dates (List[Tuple[int, int, int]]): List of training dates.
        test_dates (List[Tuple[int, int, int]]): List of testing dates.
        time_interval (int): Time interval of the data in minutes.
    """

    def __init__(self, datapath: str) -> None:
        """
        Initialize the GeneralPowerDataManager object.

        Parameters:
            datapath (str): Path to the CSV file containing the data.
        """
        if datapath is None:
            raise ValueError("Please input the correct datapath")

        data = pd.read_csv(datapath, low_memory=False)

        # Check if 'DataTime' column exists
        if 'DataTime' in data.columns:
            data.set_index('DataTime', inplace=True)
        else:
            first_col = data.columns[0]
            data.set_index(first_col, inplace=True)

        data.index = pd.to_datetime(data.index)

        # Print data scale and initialize time interval
        min_date = data.index.min()
        max_date = data.index.max()
        print(f"Data scale: from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        self.time_interval = int((data.index[1] - data.index[0]).seconds / 60)
        print(f"Data time interval: {self.time_interval} minutes")

        # Initialize other attributes
        self.df = data
        self.data_array = data.values


        # Display dataset information
        print(f"Dataset loaded from {datapath}")
        print(f"Dataset dimensions: {self.df.shape}")

        # split the train and test dates
        self.train_dates = []
        self.test_dates = []
        self.split_data_set()
        self._replace_nan()
        self._check_for_nan()


    def _replace_nan(self) -> None:
        """
        Replace NaN values in the data with interpolated values or the average of the surrounding values.
        """
        # 假设self.df是一个DataFrame
        self.df.infer_objects(copy=False)  # 将object类型的列推断为适当的类型
        self.df.interpolate(inplace=True)  # 然后进行插值
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)

    def _check_for_nan(self) -> None:
        """
        Check if any of the arrays contain NaN values and raise an error if they do
        """
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("Data still contains NaN values after preprocessing")

    def select_timeslot_data(self, year: int, month: int, day: int, timeslot: int) -> np.ndarray:
        """
               Select data for a specific timeslot on a specific day.

               Parameters:
                   year (int): The year of the date.
                   month (int): The month of the date.
                   day (int): The day of the date.
                   timeslot (int): The timeslot index.

               Returns:
                   np.ndarray: The data for the specified timeslot.
               """
        dt = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0) + pd.Timedelta(
            minutes=self.time_interval * timeslot)
        if dt not in self.df.index:
            raise ValueError(f"Timeslot {dt} not found in data")
        row = self.df.loc[dt]
        return row.values


    def select_day_data(self, year: int, month: int, day: int) -> np.ndarray:
        """
        Select data for a specific day.

        Parameters:
            year (int): The year of the date.
            month (int): The month of the date.
            day (int): The day of the date.

        Returns:
            np.ndarray: The data for the specified day.
        """
        start_dt = pd.Timestamp(year=year, month=month, day=day, hour=0, minute=0, second=0)
        end_dt = start_dt + pd.Timedelta(days=1)
        day_data = self.df.loc[start_dt:end_dt - pd.Timedelta(minutes=self.time_interval), :]
        return day_data.values

    def list_dates(self) -> List[Tuple[int, int, int]]:
        """
               List all available dates in the data.

               Returns:
                   List[Tuple[int, int, int]]: A list of available dates as (year, month, day).
               """
        dates = pd.Series(self.df.index).dropna().dt.normalize().unique()
        year_month_day = [(dt.year, dt.month, dt.day) for dt in dates]
        # dates = [(dt.year, dt.month, dt.day) for dt in self.df.index.normalize().unique()]
        # year_month_day = [(int(date[:4]), int(date[5:7]), int(date[8:10])) for date in dates]
        return year_month_day

    def random_date(self) -> Tuple[int, int, int]:
        """
                Randomly select a date from the available dates in the data.

                Returns:
                    Tuple[int, int, int]: The year, month, and day of the selected date.
                """

        dates = self.list_dates()
        year, month, day = random.choice(dates)
        return year, month, day
    def split_data_set(self):
        """
        Split the data into training and testing sets based on the date.

        The first three weeks of each month are used for training and the last week for testing.
        """
        all_dates = self.list_dates()
        all_dates.sort(key=lambda x: (x[0], x[1], x[2]))  # Sort dates

        train_dates = []
        test_dates = []

        current_month = all_dates[0][1]
        current_year = all_dates[0][0]
        monthly_dates = []

        for date in all_dates:
            year, month, day = date
            if month != current_month or year != current_year:
                # Sort monthly dates and split into train and test
                monthly_dates.sort()
                train_len = int(len(monthly_dates) * (3 / 4))  # First three weeks for training
                train_dates += monthly_dates[:train_len]
                test_dates += monthly_dates[train_len:]

                # Reset for the new month
                monthly_dates = []
                current_month = month
                current_year = year

            monthly_dates.append(date)

        # Handle the last month
        if len(monthly_dates) > 0:
            monthly_dates.sort()
            train_len = int(len(monthly_dates) * (3 / 4))
            train_dates += monthly_dates[:train_len]
            test_dates += monthly_dates[train_len:]

        self.train_dates = train_dates
        self.test_dates = test_dates

if __name__ == "__main__":
    # 设置你的CSV文件路径
    csv_path = "../data/reactive_power-data/IEEE123/load_reactive.csv"  # 请替换为实际CSV文件路径

    # 初始化 GeneralPowerDataManager
    print("测试初始化...")
    try:
        manager = GeneralPowerDataManager(csv_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        exit()

    # 测试1: 检查数据加载和属性初始化
    print("\n测试1: 数据加载和属性初始化")
    print(f"数据维度: {manager.df.shape}")
    print(f"时间间隔: {manager.time_interval} 分钟")


    # 测试2: 检查缺失值处理
    print("\n测试2: 缺失值处理")
    manager._replace_nan()
    has_nan = manager.df.isnull().any().any()
    print(f"是否存在NaN: {has_nan}")
    assert not has_nan, "缺失值处理失败，仍有NaN"

    # 测试3: 选择特定时间槽数据
    print("\n测试3: 选择时间槽数据")
    try:
        # 使用数据集中的第一个日期
        first_date = manager.df.index[0]
        data = manager.select_timeslot_data(first_date.year, first_date.month, first_date.day, 4)  # 第4个时间槽
        print(f"时间槽数据: {data[:5]}...")  # 只打印前5个值
        assert data.shape[0] == len(manager.df.columns), "时间槽数据维度错误"
    except Exception as e:
        print(f"时间槽选择失败: {e}")

    # 测试4: 选择全天数据
    print("\n测试4: 选择全天数据")
    try:
        day_data = manager.select_day_data(first_date.year, first_date.month, first_date.day+1)
        print(f"全天数据维度: {day_data.shape}")
        assert day_data.shape[1] == len(manager.df.columns), "全天数据列数错误"
    except Exception as e:
        print(f"全天数据选择失败: {e}")

    # 测试5: 列出所有日期
    print("\n测试5: 列出日期")
    dates = manager.list_dates()
    print(f"日期数量: {len(dates)}")
    print(f"前5个日期: {dates[:5]}")
    assert len(dates) > 0, "日期列表为空"

    # 测试6: 随机选择日期
    print("\n测试6: 随机日期")
    random.seed(42)  # 固定种子确保可复现
    random_date = manager.random_date()
    print(f"随机日期: {random_date}")
    assert random_date in dates, "随机日期不在日期列表中"

    # 测试7: 训练/测试集拆分
    print("\n测试7: 训练/测试集拆分")
    manager.split_data_set()
    print(f"训练集日期数量: {len(manager.train_dates)}")
    print(f"测试集日期数量: {len(manager.test_dates)}")
    assert len(manager.train_dates) > 0, "训练集为空"
    assert len(manager.test_dates) > 0, "测试集为空"

    # 测试8: 无效文件路径
    print("\n测试8: 无效文件路径")
    try:
        GeneralPowerDataManager("invalid_path.csv")
        print("错误：无效路径未抛出异常")
    except FileNotFoundError:
        print("无效路径测试通过")

    print("\n所有测试完成！")