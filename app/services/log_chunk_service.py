class LogChunkService:
    def chunk_logs(
        self,
        text: str,
        chunk_size: int = 1000,
    ):
        chunks = []

        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

        return chunks


log_chunk_service = LogChunkService()