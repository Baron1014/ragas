from langchain_community.document_loaders import TextLoader, PyPDFLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config
import os

# 獲取當前工作目錄的絕對路徑
current_dir = os.getcwd()
# 獲取相對路徑的絕對路徑
## TXT Loader
file_path = './taipower_demo/news/'
faq_path = './taipower_demo/taipower_FAQ/'
absolute_path = os.path.abspath(os.path.join(current_dir, file_path))
absolute_faq_path = os.path.abspath(os.path.join(current_dir, faq_path))

MAC_DOC = 150
loaders = [TextLoader(os.path.join(absolute_path, fn)) for fn in os.listdir(absolute_path)]
loaders.extend([TextLoader(os.path.join(absolute_faq_path, fn)) for fn in os.listdir(absolute_faq_path)])
documents = [loader.load()[0] for loader in loaders]

# PDF Loader
file_path = './taipower_demo/others/'
absolute_path = os.path.abspath(os.path.join(current_dir, file_path))
loaders = [PyPDFLoader(os.path.join(absolute_path, fn), extract_images=True) for fn in os.listdir(absolute_path)]
for loader in loaders:
    documents.extend(loader.load_and_split())

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4-turbo")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)
generator.adapt('zh-tw', [simple, multi_context, reasoning, conditional])
generator.save([simple, multi_context, reasoning, conditional])

# Change resulting question type distribution
distributions = {
    simple: 0.25,
    multi_context: 0.25,
    reasoning: 0.25,
    conditional:0.25
}

# use generator.generate_with_llamaindex_docs if you use llama-index as document loader
testset = generator.generate_with_langchain_docs(documents, MAC_DOC, distributions)
df = testset.to_pandas()
df.to_csv('out.csv', index=False, encoding='utf_8_sig')
