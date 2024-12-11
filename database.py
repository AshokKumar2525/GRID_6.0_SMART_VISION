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
def calculate_expected_life_span(produce, freshness):
    """
    Calculate the expected life span based on the produce and its freshness.

    :param produce: The type of produce (e.g., 'Apple', 'Banana').
    :param freshness: The freshness level (e.g., 'Fresh', 'Rotten').
    :return: The expected life span in days as a string.
    """
    # Define the mapping for expected life spans
    freshness_life_span_mapping = {
        100: {
            'Apple': 30, 'Banana': 7, 'Carrot': 21, 'Cucumber': 14, 'Pepper': 21,
            'Potato': 60, 'Tomato': 10, 'Mango': 14, 'Melon': 10, 'Orange': 30,
            'Peach': 7, 'Pear': 14
        },
        75: {
            'Apple': 15, 'Banana': 4, 'Mango': 7, 'Melon': 5, 'Orange': 15,
            'Peach': 4, 'Pear': 7
        },
        50: {
            'Apple': 5, 'Banana': 2, 'Mango': 3, 'Melon': 2, 'Orange': 5,
            'Peach': 2, 'Pear': 3
        },
        25: {
            'Apple': 0, 'Banana': 0, 'Carrot': 0, 'Cucumber': 0, 'Pepper': 0,
            'Potato': 0, 'Tomato': 0, 'Mango': 0, 'Melon': 0, 'Orange': 0,
            'Peach': 0, 'Pear': 0
        }
    }

    # Get the expected life span
    return str(freshness_life_span_mapping.get(freshness, {}).get(produce, 0))

def insert_freshness_data_batch(data_list):
    """
    Insert multiple records into the `freshness_detection` table in a single database connection.
    The `expected_life_span` is calculated based on `produce` and `freshness`.

    :param data_list: List of tuples containing (produce, freshness).
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

        # Prepare data with the timestamp and calculated expected life span
        values = [
            (
                timestamp,
                produce,
                freshness,
                calculate_expected_life_span(produce, freshness)
            )
            for produce, freshness in data_list
        ]

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

            
def calculate_expiry_details(expiry_date_str):
    """
    Calculate whether the product is expired and the expected life span (in days) from today.

    :param expiry_date_str: The expiry date in string format (e.g., '02/11/2024' or 'NA').
    :return: A tuple (expired, expected_life_span_days).
    """
    if expiry_date_str == 'NA':
        return 'NA', 'NA'  # Both expired and expected life span as 'NA'

    current_date = datetime.now()
    expiry_date = datetime.strptime(expiry_date_str, '%d/%m/%Y').date()

    # Calculate if expired
    expired = 'Yes' if expiry_date < current_date.date() else 'No'


    # Calculate expected life span in days


    expected_life_span_days = "NA" if expired == "Yes" else (expiry_date - current_date.date()).days
    return expired, str(expected_life_span_days)

def insert_brand_data_batch(data_list):
    """
    Insert multiple brand recognition records into the `brand_recognition` table.

    :param data_list: List of tuples containing (brand, count, expiry_date_str).
    """
    print("Starting insert_brand_data_batch...")

    if not data_list:
        print("No data to insert.")
        return

    try:
        timestamp = datetime.now().isoformat()  # ISO 8601 format for the timestamp
        connection = get_db_connection()  # Get the database connection

        if not connection:
            print("Failed to connect to the database.")
            return  # Exit if connection fails

        cursor = connection.cursor()  # Create a cursor

        query = """
            INSERT INTO brand_recognition (timestamp, brand, expiry_date, count, expired, expected_life_span)
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        # Prepare data with the timestamp included for each record and calculate expiry details
        values = []
        for record in data_list:
            if len(record) != 3:
                print(f"Skipping invalid record: {record}")
                continue  # Skip records that don't match the expected structure

            brand, count, expiry_date_str = record
            expired, expected_life_span = calculate_expiry_details(expiry_date_str)
            values.append((timestamp, brand, expiry_date_str, count, expired, expected_life_span))

        if not values:
            print("No valid data to insert after validation.")
            return

        cursor.executemany(query, values)  # Execute all queries in one go
        connection.commit()  # Commit the transaction

        print(f"Successfully inserted {len(values)} records into the database.")

    except psycopg2.Error as err:
        print(f"PostgreSQL Error: {err}")  # Catch PostgreSQL-specific errors
    except Exception as e:
        print(f"Unexpected error: {e}")  # Catch any other unexpected errors
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()  # Close the cursor
        if 'connection' in locals() and connection:
            connection.close()  # Close the connection

if __name__ == "__main__":
    #insert_freshness_data("mango",75,4)
    mapped_results = [
        ('Coca Cola Can 250ml', 2, 'NA'),
        ('Fanta 500ml', 2, 'NA'),
        ('Kurkure Chutney Chaska 62gm', 2, '8/07/2024'),
        ('Colgate Maximum Cavity Protection 75gm', 2, '02/11/2024'),
        ('Lays Wavy Mexican Chili 34gm', 1, '05/03/2025')
    ]

    insert_brand_data_batch(mapped_results)

    