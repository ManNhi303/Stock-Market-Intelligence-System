# Databricks notebook source
from pyspark.sql.functions import col, max

# Cấu hình kết nối Azure Data Lake Storage
data_lake_name = "storagestockstreaming"
secret = "MxU8Q~aC3vC9Pj3VyRYdNvHiUVo.Iq2WNGlM7acp"
app_id = "ca347baa-c612-47d4-97f1-e9f9577cbd57"
dir_id = "40127cd4-45f3-49a3-b05d-315a43a9f033"

spark.conf.set(f"fs.azure.account.auth.type.{data_lake_name}.dfs.core.windows.net", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{data_lake_name}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{data_lake_name}.dfs.core.windows.net", app_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{data_lake_name}.dfs.core.windows.net", secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{data_lake_name}.dfs.core.windows.net", f"https://login.microsoftonline.com/{dir_id}/oauth2/token")

# Đường dẫn đến các thư mục trên Azure Data Lake
silver = f'abfss://silver@{data_lake_name}.dfs.core.windows.net'
gold = f'abfss://gold@{data_lake_name}.dfs.core.windows.net'

# 1. Lấy `max(updated)` từ bảng `gold`
max_time_gold_report_day = (
    spark.read.format("delta")
    .load(f"{gold}/MANAGEMENT/REPORT/report_day")
    .agg(max("updated").alias("max_updated"))
    .collect()[0]["max_updated"]
)

max_time_gold_report_intraday = (
    spark.read.format("delta")
    .load(f"{gold}/MANAGEMENT/REPORT/report_intraday")
    .agg(max("updated").alias("max_updated"))
    .collect()[0]["max_updated"]
)

# 2. Đọc dữ liệu mới và ép kiểu volume thành double
df_new_day = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(f'{silver}/DAY/new_data_day_1')


df_new_intraday = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(f'{silver}/INTRADAY/new_data_intraday')


# 3. Lọc dữ liệu mới từ `silver`
df_silver_report_day = spark.read.csv(f"{silver}/MANAGEMENT/REPORT/report_day", header=True, inferSchema=True)
df_silver_report_intraday = spark.read.csv(f"{silver}/MANAGEMENT/REPORT/report_intraday", header=True, inferSchema=True)

df_silver_new_report_day = df_silver_report_day.filter(col("updated") > max_time_gold_report_day)
df_silver_new_report_intraday = df_silver_report_intraday.filter(col("updated") > max_time_gold_report_intraday)

# 4. Ghi dữ liệu mới vào `gold`
if df_silver_new_report_day.count() > 0:
    # Overwrite report
    df_silver_new_report_day.write.format("delta").mode("overwrite").option("path", f"{gold}/MANAGEMENT/REPORT/report_day").saveAsTable('gold.report_day')
    # Append day
    df_new_day.write.format("delta").mode("append").option("path", f"{gold}/Day").saveAsTable('gold.DAY')
    print("Cập nhật xong dữ liệu vào gold.report_day và gold.day.")

if df_silver_new_report_intraday.count() > 0:
    # Overwrite report
    df_silver_new_report_intraday.write.format("delta").mode("overwrite").option("path", f"{gold}/MANAGEMENT/REPORT/report_intraday").saveAsTable('gold.report_intraday')
    # Append intraday
    df_new_intraday.write.format("delta").mode("append").option("path", f"{gold}/INTRADAY").saveAsTable('gold.INTRADAY')
    print("Cập nhật xong dữ liệu vào gold.report_intraday và gold.intraday.")

# Hiển thị dữ liệu mới
df_silver_new_report_day.show(5, truncate=False)
df_silver_new_report_intraday.show(5, truncate=False)


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Tìm giá trị `updated` mới nhất trong bảng
# MAGIC SELECT *
# MAGIC FROM gold.DAY
# MAGIC WHERE date = (SELECT MAX(date) FROM gold.DAY);
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Tìm giá trị `updated` mới nhất trong bảng
# MAGIC SELECT *
# MAGIC FROM gold.DAY
# MAGIC WHERE date = (SELECT MAX(date) FROM gold.DAY);