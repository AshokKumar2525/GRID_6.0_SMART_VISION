import psycopg2
from datetime import datetime

# Database connection details
DB_URL = "postgresql://smart_vision_user:oTUhOMKcHLwQQhIjD9RtG5a4zsMEgbii@dpg-ctaiof3tq21c73c518o0-a.oregon-postgres.render.com/smart_vision"

def get_db_connection():
    try:
        print("Connecting to the database...")
        connection = psycopg2.connect(DB_URL)
        print("Connection established successfully.")
        return connection
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def insert_freshness_data_batch(data_list):
    """
    Insert multiple records into the `freshness_detection` table in a single database connection.

    :param data_list: List of tuples containing (produce, freshness, expected_life_span).
    """
    print("Starting insert_freshness_data_batch...")

    if not data_list:
        print("No data to insert.")
        return

    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current timestamp
        connection = get_db_connection()  # Get the database connection

        if not connection:
            print("Failed to connect to the database.")
            return  # Exit if connection fails

        cursor = connection.cursor()  # Create a cursor

        query = """
            INSERT INTO freshness_detection (timestamp, produce, freshness, expected_life_span)
            VALUES (%s, %s, %s, %s)
        """

        # Prepare data with the timestamp included for each record
        values = [(timestamp, produce, freshness, expected_life_span) for produce, freshness, expected_life_span in data_list]

        cursor.executemany(query, values)  # Execute all queries in one go
        connection.commit()  # Commit the transaction

        print(f"Successfully inserted {len(data_list)} records into the database.")

    except psycopg2.Error as err:
        print(f"PostgreSQL Error: {err}")  # Catch PostgreSQL-specific errors
    except Exception as e:
        print(f"Unexpected error: {e}")  # Catch any other unexpected errors
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()  # Close the cursor
        if 'connection' in locals() and connection:
            connection.close()  # Close the connection

            
def insert_brand_data(brand, expiry_date, count, expired, expected_life_span):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current timestamp
    connection = get_db_connection()  # Get the database connection
    if not connection:
        print("Failed to connect to the database.")
        return

    try:
        cursor = connection.cursor()  # Create a cursor
        query = """
            INSERT INTO brand_recognition (timestamp, brand, expiry_date, count, expired, expected_life_span)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (timestamp, brand, expiry_date, count, expired, expected_life_span)
        cursor.execute(query, values)  # Execute the query with values
        connection.commit()  # Commit the transaction
        print("Data inserted successfully.")
    except psycopg2.Error as err:
        print(f"MySQL Error: {err}")  # Catch MySQL-specific errors
    except Exception as e:
        print(f"Unexpected error: {e}")  # Catch any other unexpected errors
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()  # Close the cursor
        if 'connection' in locals() and connection:
            connection.close()  # Close the connection


if __name__ == "__main__":
    insert_freshness_data("mango",75,4)
    