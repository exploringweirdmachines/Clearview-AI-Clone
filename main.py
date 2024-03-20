import argparse
import atexit
import sys
import box
import yaml
import pathlib
import logging.config
import logging.handlers
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import MultiModalRetriever
from haystack import Document

logger = logging.getLogger(__name__)

with open('configs/config.yml', 'r', encoding='utf8') as ymlfile:
    config = box.Box(yaml.safe_load(ymlfile))


def setup_logging():
    config_file = pathlib.Path("configs/log_config.yml")
    with open(config_file) as log_file:
        log_config = yaml.safe_load(log_file)
    logging.config.dictConfig(log_config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="Image Finder",
        description="Tool which finds images given an image as input",
        epilog="Thanks for using {prog}"
    )
    parser.add_argument("-v", "--version", dest="version", action="version", version=f"%(prog)% {config.version}")

    subparsers = parser.add_subparsers(dest="command", title="Commands", metavar="COMMAND")

    create_db_parser = subparsers.add_parser("create_db", help="Create a new FAISS database")
    create_db_parser.add_argument("-f", "--files", type=str, required=True, help="Path to the folder containing images", default="examples/images")
    create_db_parser.add_argument("-o", "--output", type=str, required=True,
                                  help="Output path of the faiss database files", default="examples/vector_database")

    load_db_parser = subparsers.add_parser("search_db", help="Search a database for an image")
    load_db_parser.add_argument("-d", "--db_path", type=str, required=True, help="Path to the FAISS database", default="examples/vector_database")
    load_db_parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the target image")

    return parser


def create_db(folder_path, images_path):
    """Create a new FAISS database or load an existing one."""
    db_path = pathlib.Path.cwd() / folder_path
    index_path = pathlib.Path(folder_path) / "index.faiss"
    config_path = pathlib.Path(folder_path) / "config.json"

    logger.debug(f"Database path: {db_path}")
    logger.debug(f"Index path: {index_path}")
    logger.debug(f"Config path: {config_path}")

    if db_path.exists():
        try:
            document_store = FAISSDocumentStore.load(index_path=str(index_path), config_path=str(config_path))
            logger.info("Existing FAISS database loaded.")
        except Exception as e:
            logger.warning(f"Error loading existing FAISS database. Reason: {str(e)}")
            document_store = None

    if document_store is None:
        logger.info("Creating a new FAISS database.")
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///{db_path}/faiss_db.db",
            faiss_index_factory_str="Flat",
            index="image_files",
            embedding_dim=512,
            similarity="cosine",
            embedding_field="meta",
        )

    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    existing_images = set()

    for file_path in pathlib.Path(images_path).glob("*"):
        if file_path.suffix.lower() in image_extensions:
            existing_images.add(str(file_path))
            try:
                image = Document(
                    content=str(file_path),
                    content_type="image",
                    meta={"filename": file_path.name},
                )
                document_store.write_documents([image])
            except Exception as e:
                logger.error(f"Error processing file {file_path}. Reason: {str(e)}")

    # Delete embeddings for non-existing image files
    all_documents = document_store.get_all_documents()
    for doc in all_documents:
        if doc.content not in existing_images:
            logger.debug(f"Deleting embeddings for non-existing image: {doc.content}")
            document_store.delete_documents(ids=[doc.id])

    document_store.update_embeddings(retriever=get_multimodal_retriever(document_store),
                                     update_existing_embeddings=False)
    document_store.save(index_path=str(index_path), config_path=str(config_path))
    logger.debug("FAISS database created/updated successfully.")


def load_db(db_path):
    try:
        logger.info(f"Loading database from: {db_path}")
        document_store = FAISSDocumentStore.load(index_path=f"{db_path}/index.faiss", config_path=f"{db_path}/config.json",)
        logger.info("FAISS database loaded successfully.")
        return document_store
    except Exception as e:
        logger.error(f"Error loading FAISS database. Reason: {str(e)}")
        return None


def get_multimodal_retriever(document_store):
    logger.debug("Loading images retriever...")
    retriever_text_to_image = MultiModalRetriever(
        document_store=document_store,
        query_embedding_model="sentence-transformers/clip-ViT-B-32",
        query_type="image",
        document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
        top_k=3,
        similarity_function="cosine",
        devices=[config.multimodalretriever.devices],
    )
    logger.debug("Images retriever loaded successfully.")
    return retriever_text_to_image


def search_images(image_path, db_path):
    document_store = load_db(db_path)
    if document_store is None:
        logger.error("No FAISS database loaded. Please load or create a database first.")
        return

    retriever = get_multimodal_retriever(document_store)
    img_path = pathlib.Path.cwd() / image_path
    logger.info(f"Searching for image: {img_path}")
    similar_images = retriever.retrieve(query=str(img_path), query_type="image", document_store=document_store)
    logger.info("TOP 3 - Found similar images:")

    for image in similar_images:
        logger.info(f"Score: {round(image.score*100, 2)}%; Image: {image.meta['filename']}")


def main():
    setup_logging()
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "create_db":
        create_db(args.output, args.files)
    elif args.command == "search_db":
        search_images(pathlib.Path(args.image_path), args.db_path)
    else:
        parser.print_help()


if __name__ == '__main__':
    #logging.basicConfig(level="INFO")
    main()