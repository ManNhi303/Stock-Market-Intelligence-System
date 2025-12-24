# Databricks notebook source
## đã cập nhật lại theo data lake mới 
data_lake_name = "storagestockstreaming"
secret = "MxU8Q~aC3vC9Pj3VyRYdNvHiUVo.Iq2WNGlM7acp"
app_id = "ca347baa-c612-47d4-97f1-e9f9577cbd57"
dir_id = "40127cd4-45f3-49a3-b05d-315a43a9f033"

# COMMAND ----------

spark.conf.set(f"fs.azure.account.auth.type.{data_lake_name}.dfs.core.windows.net", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{data_lake_name}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{data_lake_name}.dfs.core.windows.net", app_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{data_lake_name}.dfs.core.windows.net", secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{data_lake_name}.dfs.core.windows.net", f"https://login.microsoftonline.com/{dir_id}/oauth2/token")


# COMMAND ----------

silver = f'abfss://silver@{data_lake_name}.dfs.core.windows.net'
gold = f'abfss://gold@{data_lake_name}.dfs.core.windows.net'
bronze = f'abfss://bronze@{data_lake_name}.dfs.core.windows.net'

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. IMPORT LIBRARIES

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType, DateType,IntegerType,TimestampType
from pyspark.sql import SparkSession


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. READ DATA

# COMMAND ----------


schema_day = StructType([
    StructField("code", StringType(), True),
    StructField("date", DateType(), True),
    StructField("open", FloatType(), True),
    StructField("high", FloatType(), True),
    StructField("low", FloatType(), True),
    StructField("close", FloatType(), True),
    StructField("adj_close", FloatType(), True),
    StructField("volume", FloatType(), True),
])


# COMMAND ----------


schema_intraday = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("open", FloatType(), True),
    StructField("high", FloatType(), True),
    StructField("low", FloatType(), True),
    StructField("close", FloatType(), True),
    StructField("volume", FloatType(), True), 
    StructField("code", StringType(), True),
    StructField("timestamp_vn", TimestampType(), True),

])


# COMMAND ----------

# Định nghĩa schema cho df_report_day
schema_report = StructType([
    StructField("code", StringType(), True),
    StructField("nb", IntegerType(), True),
    StructField("min_date", DateType(), True),
    StructField("max_date", DateType(), True),
    StructField("updated", TimestampType(), True)
])

# COMMAND ----------

## Khởi tạo SparkSession
spark = SparkSession.builder.getOrCreate()

# Đọc dữ liệu từ thư mục RAW (dữ liệu mới vừa cào về)
df_new_day = spark.read.format("csv")\
    .option("header", "true")\
    .schema(schema_day)\
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/DAY/new_data_day.csv")

df_new_intraday = spark.read.format("csv")\
    .option("header", "true")\
    .schema(schema_intraday)\
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/INTRADAY/new_data_intraday.csv")

# Đọc dữ liệu các mã chứng khoán và báo cáo từ MANAGEMENT
df_list_code = spark.read.format("csv")\
    .option("header", "true")\
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/LIST_CODE.csv")

df_report_day = spark.read.format("csv")\
    .option("header", "true")\
    .schema(schema_report)\
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/report_day.csv")

df_report_intraday = spark.read.format("csv")\
    .option("header", "true")\
    .schema(schema_report)\
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/report_intraday.csv")

# Đọc dữ liệu từ SILVER để kiểm tra updated
df_silver_report_day = spark.read.format("csv")\
    .option("header", "true")\
    .schema(schema_report)\
    .load(f"abfss://silver@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/REPORT/report_day")

df_silver_report_intraday = spark.read.format("csv")\
    .option("header", "true")\
    .schema(schema_report)\
    .load(f"abfss://silver@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/REPORT/report_intraday")



# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Hàm kiểm tra dữ liệu mới cần xử lý
def filter_new_data(df_report_bronze, df_report_silver):
    # Lấy dòng đầu tiên từ mỗi bảng
    bronze_first_row = df_report_bronze.orderBy("updated").first()
    silver_first_row = df_report_silver.orderBy("updated").first()

    # So sánh thời gian cập nhật
    if bronze_first_row and silver_first_row:
        return bronze_first_row['updated'] > silver_first_row['updated']
    elif bronze_first_row and not silver_first_row:
        return True
    return False

# Hàm transform dữ liệu với xử lý null và trùng lặp
def transform_data_day(df, df_list_code):
    window_spec = Window.partitionBy("code").orderBy("date")
    # Xử lý null values
    df = df.withColumn("open", coalesce(col("open"), lag("open", 1).over(window_spec))) \
           .withColumn("high", coalesce(col("high"), lag("high", 1).over(window_spec))) \
           .withColumn("low", coalesce(col("low"), lag("low", 1).over(window_spec))) \
           .withColumn("close", coalesce(col("close"), lag("close", 1).over(window_spec))) \
           .withColumn("adj_close", coalesce(col("adj_close"), lag("adj_close", 1).over(window_spec)))
    # Loại bỏ dữ liệu trùng lặp
    df = df.dropDuplicates()

    # Kết hợp với danh sách mã chứng khoán
    df = df.join(df_list_code, on="code", how="left")
    return df
# Hàm transform dữ liệu với xử lý null và trùng lặp
def transform_data_intraday(df, df_list_code):
    window_spec = Window.partitionBy("code").orderBy("timestamp_vn")

    # Xử lý null values
    df = df.withColumn("open", coalesce(col("open"), lag("open", 1).over(window_spec))) \
           .withColumn("high", coalesce(col("high"), lag("high", 1).over(window_spec))) \
           .withColumn("low", coalesce(col("low"), lag("low", 1).over(window_spec))) \
           .withColumn("close", coalesce(col("close"), lag("close", 1).over(window_spec))) 
    # Loại bỏ dữ liệu trùng lặp
    df = df.dropDuplicates()

    # Kết hợp với danh sách mã chứng khoán
    df = df.join(df_list_code, on="code", how="left")
    return df

# Xử lý dữ liệu DAY
if filter_new_data(df_report_day, df_silver_report_day):
    print("Cập nhật dữ liệu DAY...")
    df_list_code_selected = df_list_code.select("code", "type")
    df_transformed_day = transform_data_day(df_new_day, df_list_code_selected)

    # Ghi dữ liệu transform vào SILVER
    df_transformed_day.write.format("csv").mode("overwrite").option("header", "true") \
        .save(f"{silver}/DAY/new_data_day_1")

    # Ghi đè file report_day
    df_report_day.write.format("csv").mode("overwrite").option("header", "true") \
        .save(f"{silver}/MANAGEMENT/REPORT/report_day")

# Xử lý dữ liệu INTRADAY
if filter_new_data(df_report_intraday, df_silver_report_intraday):
    print("Cập nhật dữ liệu INTRADAY...")
    df_list_code_selected = df_list_code.select("code", "type")
    df_transformed_intraday = transform_data_intraday(df_new_intraday, df_list_code_selected)

    # Ghi dữ liệu transform vào SILVER
    df_transformed_intraday.write.format("csv").mode("overwrite").option("header", "true") \
        .save(f"{silver}/INTRADAY/new_data_intraday")

    # Ghi đè file report_intraday
    df_report_intraday.write.format("csv").mode("overwrite").option("header", "true") \
        .save(f"{silver}/MANAGEMENT/REPORT/report_intraday") 
