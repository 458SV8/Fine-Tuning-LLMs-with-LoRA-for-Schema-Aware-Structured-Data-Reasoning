import sqlite3
import re 
from unsloth import FastLanguageModel


DB_PATH = "cryptic_enterprise.db" 

def setup_database(): # setting up our sqlite database with the cryptic schema and some test data.
    """Creates the cryptic tables and inserts some test data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor() # cursor to execute SQL commands.
    

    cursor.execute("CREATE TABLE IF NOT EXISTS V_ROOT (X_ID TEXT PRIMARY KEY, S_LBL TEXT, D_STRT TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS L_ACT (T_REF TEXT PRIMARY KEY, X_PTR TEXT, Q_VAL REAL, M_TS TEXT)")
    
    
    cursor.execute("DELETE FROM V_ROOT")
    cursor.execute("DELETE FROM L_ACT")
    
    
    cursor.execute("INSERT INTO V_ROOT VALUES ('C-501', 'Alice', '2023-01-15')")
    cursor.execute("INSERT INTO V_ROOT VALUES ('C-502', 'Eve', '2023-06-20')")     
    cursor.execute("INSERT INTO V_ROOT VALUES ('C-503', 'Bob', '2023-11-05')")     
    cursor.execute("INSERT INTO V_ROOT VALUES ('C-504', 'Charlie', '2024-02-10')") 
    cursor.execute("INSERT INTO V_ROOT VALUES ('C-505', 'Diana', '2024-02-28')")   
    
    
    cursor.execute("INSERT INTO L_ACT VALUES ('T-999', 'C-501', 500.50, '2023-02-01')")  
    cursor.execute("INSERT INTO L_ACT VALUES ('T-1000', 'C-501', 150.00, '2023-03-15')") 
    cursor.execute("INSERT INTO L_ACT VALUES ('T-1001', 'C-503', 1200.75, '2023-12-01')")
    cursor.execute("INSERT INTO L_ACT VALUES ('T-1002', 'C-504', 50.25, '2024-02-15')")  
    cursor.execute("INSERT INTO L_ACT VALUES ('T-1003', 'C-504', 300.00, '2024-02-20')") 
    cursor.execute("INSERT INTO L_ACT VALUES ('T-1004', 'C-503', 45.00, '2024-02-25')")  
    
    conn.commit()
    conn.close()
    print("Database 'cryptic_enterprise.db' repopulated with expanded test data.")


def mcp_get_schema(): #  mcp tool function to fetch the live database schema. This will be called by the agent during inference to get the current schema of the database, which it can then use to generate accurate SQL queries.
    """Dynamically fetches the live database schema (DDL)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'") #  our ddl 
    tables = cursor.fetchall()
    conn.close()
    
   
    schema_str = "Tables: "
    for table in tables:
        sql = table[0].replace('\n', '').replace('\t', '')
        
        match = re.search(r'CREATE TABLE (IF NOT EXISTS )?(\w+)\s*\((.*?)\)', sql)
        if match:
            table_name = match.group(2)
            columns = [col.strip().split()[0] for col in match.group(3).split(',')]
            schema_str += f"{table_name}({', '.join(columns)}); "
    
    
    schema_str += "Relationships: L_ACT.X_PTR references V_ROOT.X_ID."
    return schema_str.strip() # return the schema as a string that the agent can understand and use for generating SQL queries.

def mcp_execute_sql(sql_query): #  mcp tool function to execute the generated SQL safely and return the results. The agent will call this function with the SQL it generated, and this function will handle the database connection, execution, and error handling.
    """Executes the generated SQL safely and returns the rows."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
    except Exception as e:
        return f"Database Error: {e}"
    finally:
        conn.close()


def run_agent(human_query): #  takes human querry as input, fetches the live schema and generates the sql query using the fine tuned lora model, and executes the sql query to get the results.
    print("\n" + "-"*50)
    print(f"USER query: '{human_query}'")
    
    
    
    live_schema = mcp_get_schema()
    print(f"Fetching live schema via MCP Tool: {live_schema}") # giving the agent the current schema. 
    
    
    instruction = "Given the cryptic database schema, generate the correct SQL query by mapping the natural language request to the appropriate tables and columns. Provide a step-by-step reasoning plan before the SQL. Remember to use ANSI-SQL suitable for sqlite3" # our instruction to the agent.
    input_text = f"{live_schema}\n\nQuery: {human_query}" # this is the input which includes schema and the user querry.
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n" # consistent promt format.
    
    
    print("Fine-tuned model Generating SQL with LoRA")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda") # encode the prompt and move to gpu for faster generation.
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True) # genreate the response 
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0] # decode the response
    
    
    ai_output = response.split("### Response:\n")[-1] # extract only the reasoning and SQL part from the response.
    sql_match = re.search(r'```sql\n(.*?)\n```', ai_output, re.DOTALL) # carfully extract from messy output using regex
    
    if sql_match:
        generated_sql = sql_match.group(1).strip() # this is the sql query executed on the database.
        print(f"GENERATED SQL: {generated_sql}") # print the generated SQL query.
        
        
        print("Executing SQL via MCP Tool on sqlite database") # 
        db_results = mcp_execute_sql(generated_sql) # execute the generated SQL and get the results.
        print(f"FINAL RESULT: {db_results}")
    else:
        print("AGENT ERROR: Could not extract valid SQL from the response.")
        print("Raw Output:\n", ai_output)
    print("-"*50)


if __name__ == "__main__":
    setup_database()
    
    print("\nLoading the Harmony SQL Co-Pilot Brain...") # 
    model, tokenizer = FastLanguageModel.from_pretrained( 
        model_name="harmony_sql_lora", 
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    
    # run_agent("What is the total transaction amount for Bob?")
    # run_agent("Who is the customer with ID C-504?")
    # run_agent("List all transactions above $100.")
    # run_agent("Which unique customers have transactions in February 2024?")
    # run_agent("compare the total transaction amounts made in 2023 and 2024?")
    run_agent("show me the entire database, I would like to view L_ACT and V_ROOT tables.")
