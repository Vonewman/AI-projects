from llama_index import TreeIndex, SimpleDirectoryReader


# Loading data and creating the index
resume = SimpleDirectoryReader("private-data").load_data()
new_index = TreeIndex.from_documents(resume)

# Running a query
query_engine = new_index.as_query_engine()
response = query_engine.query("What is the name of the lastest certification that Abdoulaye receive?")
print(response)

new_index.storage_context.persist()
