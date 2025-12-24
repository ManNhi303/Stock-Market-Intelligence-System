# Databricks notebook source
spark.conf.set("fs.azure.account.key.storagestockstreaming.dfs.core.windows.net", "Piyq2P+pcRLLMYkt9DewtfrXe1qaOPwbmxKRePTW8IKzATHeIUdXkX6ww8ga5yLdNS9AgwGYa1Nm+ASt42LHlQ==")


# COMMAND ----------

# MAGIC %md
# MAGIC # **DATA ACCESS**
# MAGIC

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

silver = f'abfss://silver@{data_lake_name}.dfs.core.windows.net'
gold = f'abfss://gold@{data_lake_name}.dfs.core.windows.net'

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC # **DATA READING, WRITING AND CREATE DELTA TABLE**

# COMMAND ----------

df_day = spark.read.format("csv")\
                .option("header", "true")\
                .option("inferSchema", "true")\
                .load(f'{silver}/DAY/data_day')

# COMMAND ----------

display(df_day)

# COMMAND ----------

df_intraday = spark.read.format("csv")\
                .option("header", "true")\
                .option("inferSchema", "true")\
                .load(f'{silver}/INTRADAY/data_intraday')

# COMMAND ----------

display(df_intraday)

# COMMAND ----------

df_list_code = spark.read.format("csv")\
                .option("header", "true")\
                .option("inferSchema", "true")\
                .load(f'{silver}/MANAGEMENT/LIST_CODE') 

# COMMAND ----------

display(df_list_code)

# COMMAND ----------

df_report_day = spark.read.format("csv")\
                .option("header", "true")\
                .option("inferSchema", "true")\
                .load(f'{silver}/MANAGEMENT/REPORT/report_day') 

# COMMAND ----------

display(df_report_day)

# COMMAND ----------

df_report_intraday = spark.read.format("csv")\
                .option("header", "true")\
                .option("inferSchema", "true")\
                .load(f'{silver}/MANAGEMENT/REPORT/report_intraday') 

# COMMAND ----------

display(df_report_intraday)

# COMMAND ----------

# MAGIC %md
# MAGIC # **CREATE DATABASE**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS gold

# COMMAND ----------

df_day.write.format('delta')\
            .mode('overwrite')\
            .option('path', f'{gold}/DAY/')\
            .saveAsTable('gold.DAY')

# COMMAND ----------

df_intraday.write.format('delta')\
            .mode('overwrite')\
            .option('path', f'{gold}/INTRADAY/')\
            .saveAsTable('gold.INTRADAY')

# COMMAND ----------

df_report_day.write.format('delta')\
            .mode('overwrite')\
            .option('path', f'{gold}/MANAGEMENT/REPORT/report_day')\
            .saveAsTable('gold.report_day')

# COMMAND ----------

df_report_intraday.write.format('delta')\
            .mode('overwrite')\
            .option('path', f'{gold}/MANAGEMENT/REPORT/report_intraday')\
            .saveAsTable('gold.report_intraday')

# COMMAND ----------

df_list_code.write.format('delta')\
            .mode('overwrite')\
            .option('path', f'{gold}/MANAGEMENT/LIST_CODE')\
            .saveAsTable('gold.LIST_CODE')

# COMMAND ----------

row_count = df_day.count()
print(f"Số lượng hàng: {row_count}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Truy vấn số dòng trong bảng gold.day
# MAGIC SELECT COUNT(*) AS row_count
# MAGIC FROM gold.intraday;
# MAGIC

# COMMAND ----------

from pyspark.sql.utils import AnalysisException

try:
    # Drop bảng
    spark.sql("DROP TABLE IF EXISTS gold.day")

    # Xóa dữ liệu vật lý nếu cần
    dbutils.fs.rm(f'{gold}/DAY', True)  # Với Databricks
except AnalysisException as e:
    print(f"Error: {e}")


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Truy vấn số dòng trong bảng gold.day
# MAGIC SELECT COUNT(*) AS row_count
# MAGIC FROM gold.intraday;

# COMMAND ----------

# MAGIC %md
# MAGIC ### **DAY DATA**

# COMMAND ----------

df_report_combined = spark.read.format("csv")\
                .option("inferSchema", "true")\
                .load(f'{silver}/MANAGEMENT/REPORT')

# COMMAND ----------

df_report_combined.display()

# COMMAND ----------

df_list_code.display()