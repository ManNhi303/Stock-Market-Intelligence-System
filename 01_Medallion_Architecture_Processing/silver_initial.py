# Databricks notebook source
# MAGIC %md
# MAGIC # 1. ACCESS BRONZE LAYER

# COMMAND ----------

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

dbutils.fs.ls(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/")

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

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. READ DATA

# COMMAND ----------


schema = StructType([
    StructField("code", StringType(), True),
    StructField("date", DateType(), True),
    StructField("open", FloatType(), True),
    StructField("high", FloatType(), True),
    StructField("low", FloatType(), True),
    StructField("close", FloatType(), True),
    StructField("adj_close", FloatType(), True),
    StructField("volume", DoubleType(), True),
])


# COMMAND ----------

df_day = spark.read.format("csv")\
                .option("header", "true")\
                .schema(schema)\
                .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/DAY/data_day.csv")
                # .option("recursiveFileLook", True)\
                # .schema(my_schema)\
df_day.inputFiles()

# COMMAND ----------

df_day.printSchema()


# COMMAND ----------

display(df_day)

# COMMAND ----------


schema_intraday = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("open", FloatType(), True),
    StructField("high", FloatType(), True),
    StructField("low", FloatType(), True),
    StructField("close", FloatType(), True),
    StructField("volume", DoubleType(), True), 
    StructField("code", StringType(), True),
    StructField("timestamp_vn", TimestampType(), True),

])


# COMMAND ----------

df_intraday = spark.read.format("csv")\
                .option("header", "true")\
                .schema(schema_intraday)\
                .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/INTRADAY/data_intraday.csv")
                # .option("recursiveFileLook", True)\

df_intraday.inputFiles()


# COMMAND ----------

df_intraday.printSchema()

# COMMAND ----------

display(df_intraday)

# COMMAND ----------

df_list_code = spark.read.format("csv") \
    .option("header", "true") \
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/LIST_CODE.csv")


# COMMAND ----------

df_list_code.printSchema()

# COMMAND ----------

display(df_list_code)

# COMMAND ----------

# Định nghĩa schema cho df_report_day
schema_report_day = StructType([
    StructField("code", StringType(), True),
    StructField("nb", IntegerType(), True),
    StructField("min_date", DateType(), True),
    StructField("max_date", DateType(), True),
    StructField("updated", TimestampType(), True)
])

# Đọc file và áp dụng schema
df_report_day = spark.read.format("csv") \
    .option("header", "true") \
    .schema(schema_report_day) \
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/report_day.csv")

# COMMAND ----------

df_report_day.printSchema()

# COMMAND ----------

display(df_report_day)

# COMMAND ----------

# Định nghĩa schema cho df_report_day
schema_report_intraday = StructType([
    StructField("code", StringType(), True),
    StructField("nb", IntegerType(), True),
    StructField("min_date", DateType(), True),
    StructField("max_date", DateType(), True),
    StructField("updated", TimestampType(), True)
])

# Đọc file và áp dụng schema
df_report_intraday = spark.read.format("csv") \
    .option("header", "true") \
    .schema(schema_report_intraday) \
    .load(f"abfss://bronze@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/report_intraday.csv")

# COMMAND ----------

df_report_intraday.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. CLEAN AND TRANSFORM DATA

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1) Thêm cột "type"

# COMMAND ----------

# Chỉ chọn các cột cần thiết: "code" và "type"
df_list_code_selected = df_list_code.select("code", "type")

# Join để thêm cột "type" từ df_list_code_selected
df_day = df_day.join(df_list_code_selected, on="code", how="left")

# Kiểm tra kết quả
df_day.show()

# COMMAND ----------

# Chỉ chọn các cột cần thiết: "code" và "type"
df_list_code_selected = df_list_code.select("code", "type")
# Join để thêm cột "type" từ df_list_code_selected
df_intraday = df_intraday.join(df_list_code_selected, on="code", how="left")

# Kiểm tra kết quả
df_intraday.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4) Loại bỏ giá trị trùng lặp

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import count, when

# Kiểm tra dữ liệu thiếu (null) trong các cột
null_count = df_day.select([ 
    (count(when(F.col(c).isNull(), c))).alias(c) 
    for c in df_day.columns
])

null_count.show()




# COMMAND ----------

# Kiểm tra dữ liệu trùng lặp
duplicate_count = df_day.count() - df_day.dropDuplicates().count()

print(f"Số lượng bản ghi trùng lặp: {duplicate_count}")

# COMMAND ----------

df_day = df_day.dropDuplicates(["code", "date"])

# COMMAND ----------

df_intraday = df_intraday.dropDuplicates(["code", "timestamp"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### (5) Kiểm tra và điền dữ liệu thiếu

# COMMAND ----------

missing_values = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
missing_values.show()

# COMMAND ----------

# Bước 1: Sắp xếp theo thứ tự tăng dần của 'code' và 'date'
df = df.orderBy(["code", "date"])

# Bước 2: Định nghĩa cửa sổ phân vùng (window spec) theo 'code' và sắp xếp theo 'date'
window_spec = Window.partitionBy("code").orderBy("date")

# Bước 3: Sử dụng lag để lấy giá trị của ngày hôm trước
df = df.withColumn("open_filled", F.coalesce(
    F.col("open"), F.lag("open", 1).over(window_spec)))\
    .withColumn("high_filled", F.coalesce(
    F.col("high"), F.lag("high", 1).over(window_spec)))\
    .withColumn("low_filled", F.coalesce(
    F.col("low"), F.lag("low", 1).over(window_spec)))\
    .withColumn("close_filled", F.coalesce(
    F.col("close"), F.lag("close", 1).over(window_spec)))\
    .withColumn("adj_close_filled", F.coalesce(
    F.col("adj_close"), F.lag("adj_close", 1).over(window_spec)))

# Bước 4: Chỉ giữ lại các cột đã điền giá trị và loại bỏ các cột gốc
df_filled = df.select(
    "code", "date", "open_filled", "high_filled", "low_filled", 
    "close_filled", "adj_close_filled", "volume", "type"
)

# Bước 5: Đổi tên cột về tên cột gốc
df_filled = df_filled.withColumnRenamed("open_filled", "open")\
    .withColumnRenamed("high_filled", "high")\
    .withColumnRenamed("low_filled", "low")\
    .withColumnRenamed("close_filled", "close")\
    .withColumnRenamed("adj_close_filled", "adj_close")

# COMMAND ----------

# MAGIC %md
# MAGIC ### (6) Kiểm tra và xử lý giá trị ngoại lai

# COMMAND ----------

# MAGIC %md
# MAGIC ### (7) Kiểm tra định dạng cột "date"

# COMMAND ----------

df_filled.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. WRITE DATA TO SILVER LAYER

# COMMAND ----------

df_day.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"abfss://silver@{data_lake_name}.dfs.core.windows.net/DAY/data_day")


# COMMAND ----------

df_intraday.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"abfss://silver@{data_lake_name}.dfs.core.windows.net/INTRADAY/data_intraday")

# COMMAND ----------

# Lưu df_report_combined vào thư mục con riêng
df_report_day.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"abfss://silver@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/REPORT/report_day")


# COMMAND ----------

# Lưu df_report_combined vào thư mục con riêng
df_report_intraday.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"abfss://silver@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/REPORT/report_intraday")




# COMMAND ----------

# Lưu df_list_code vào thư mục con riêng
df_list_code.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"abfss://silver@{data_lake_name}.dfs.core.windows.net/MANAGEMENT/REPORT/report_day")