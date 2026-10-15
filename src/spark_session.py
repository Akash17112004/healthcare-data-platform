"""Creates the local Spark session used by every stage."""
from pyspark.sql import SparkSession


def get_spark(app_name: str = "SmartHealthcarePlatform") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "Asia/Kolkata")
        # malformed text casts to NULL instead of raising; Silver handles those nulls explicitly
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
