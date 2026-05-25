import pandas as pd

# Define the columns since the .txt file doesn't have a header row
columns = [
    "Sense_ID", 
    "Primary_Descriptor", 
    "Secondary_Descriptor", 
    "Lemgram", 
    "Lemma", 
    "POS", 
    "Paradigm"
]

# Read the .txt file as a Tab-Separated Value file
df = pd.read_csv("saldo_2.3/saldo20v03.txt", sep="\t", names=columns)

# Now you can easily look up the primary meaning of a word
print(df[df['Lemma'] == '1984'])