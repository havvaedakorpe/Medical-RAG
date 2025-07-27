from Bio import Entrez
import json

#set the email
Entrez.email = "korpe20@itu.edu.tr"


terms = ["diabetes", "cancer", "hypertension", "asthma", "COVID-19"] #medical search terms
query = " OR ".join(f"({term})" for term in terms) #combine terms

#search and return 1000 articles from PubMed
handle = Entrez.esearch(db="pubmed", term=query, retmax=1000)
record = Entrez.read(handle)
id_list = record["IdList"]
#fetch the abstracts for the retrieved articles
handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
records = Entrez.read(handle)

#extract abstract text from each article
abstracts = []
for article in records['PubmedArticle']:
    try:
        abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText']
        if isinstance(abstract, list):
            abstract_text = " ".join(str(x) for x in abstract)
        else:
            abstract_text = str(abstract)
    except KeyError:
        abstract_text = ""
    abstracts.append({"content": abstract_text})
    
#save the abstracts to a JSON file
with open("pubmed_abstracts.json", "w", encoding="utf-8") as f:
    json.dump(abstracts, f, ensure_ascii=False, indent=2)
    
print("Abstracts are saved.")
