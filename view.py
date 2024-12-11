import psycopg2

# Database connection details
DB_URL = "postgresql://smart_vision_user:oTUhOMKcHLwQQhIjD9RtG5a4zsMEgbii@dpg-ctaiof3tq21c73c518o0-a.oregon-postgres.render.com/smart_vision"
def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    try:
        print("Connecting to the database...")
        connection = psycopg2.connect(DB_URL)
        print("Connection established successfully.")
        return connection
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def fetch_data_from_database():
    """
    Fetch data from brand_recognition and freshness_detection tables.
    """
    try:
        connection = get_db_connection()

        if not connection:
            return [], []  # Return empty lists if connection fails

        cursor = connection.cursor()

        # Query brand_recognition table
        cursor.execute("SELECT * FROM brand_recognition;")
        brand_data = cursor.fetchall()

        # Query freshness_detection table
        cursor.execute("SELECT * FROM freshness_detection;")
        freshness_data = cursor.fetchall()

        return brand_data, freshness_data

    except psycopg2.Error as e:
        print(f"Error fetching data: {e}")
        return [], []  # Return empty lists in case of error

    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()

def view_data():
    try:
        print("Connecting to the database...")
        connection = psycopg2.connect(DB_URL)
        cursor = connection.cursor()

        # Query brand_recognition table
        print("\nData from brand_recognition:")
        cursor.execute("SELECT * FROM brand_recognition;")
        brand_data = cursor.fetchall()
        for row in brand_data:
            print(row)

        # Query freshness_detection table
        # print("\nData from freshness_detection:")
        # cursor.execute("SELECT * FROM freshness_detection;")
        # freshness_data = cursor.fetchall()
        # for row in freshness_data:
        #     print(row)

    except psycopg2.Error as e:
        print(f"Error viewing data: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()

def delete_data_from_table(table_name, n):
    """
    Delete the first n rows from the specified table. If n is greater than the
    total number of rows in the table, delete all rows.
    """
    try:
        connection = get_db_connection()

        if not connection:
            return

        cursor = connection.cursor()

        # Check the number of rows in the table
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]

        # If there are fewer rows than n, delete all rows
        rows_to_delete = min(n, row_count)

        # Delete rows
        cursor.execute(f"DELETE FROM {table_name} WHERE ctid IN (SELECT ctid FROM {table_name} LIMIT {rows_to_delete});")
        connection.commit()
        print(f"Successfully deleted {rows_to_delete} rows from the {table_name} table.")

    except psycopg2.Error as e:
        print(f"Error deleting data: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()

def get_table_choice_and_rows():
    """
    Display a menu to select the table and prompt for the number of rows to delete.
    """
    print("Choose the table to delete data from:")
    print("1. Brand Recognition (brand_recognition)")
    print("2. Freshness Detection (freshness_detection)")
    choice = input("Enter the number corresponding to your choice: ")

    if choice == "1":
        table_name = "brand_recognition"
    elif choice == "2":
        table_name = "freshness_detection"
    else:
        print("Invalid choice. Please select a valid option.")
        return None, None

    try:
        n = int(input("Enter the number of rows to delete: "))
        return table_name, n
    except ValueError:
        print("Invalid number. Please enter a valid integer for the number of rows.")
        return None, None

if __name__ == "__main__":
    table_name, n = get_table_choice_and_rows()
    if table_name and n is not None:
        delete_data_from_table(table_name, n)


