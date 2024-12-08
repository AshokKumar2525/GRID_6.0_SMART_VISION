import psycopg2

# Database connection details
DB_URL = "postgresql://smart_vision_user:oTUhOMKcHLwQQhIjD9RtG5a4zsMEgbii@dpg-ctaiof3tq21c73c518o0-a.oregon-postgres.render.com/smart_vision"

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
        print("\nData from freshness_detection:")
        cursor.execute("SELECT * FROM freshness_detection;")
        freshness_data = cursor.fetchall()
        for row in freshness_data:
            print(row)

    except psycopg2.Error as e:
        print(f"Error viewing data: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()

if __name__ == "__main__":
    view_data()
