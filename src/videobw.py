import numpy as np
import av

'''
Converte RGB para valor de luminância
(bw) usando os valores default.
'''
def luminance(img: np.ndarray) -> np.ndarray:
    lum = [0.2126, 0.7152, 0.0722]
    return np.dot(img[..., :3], lum).astype(np.uint8)


'''
Representação de um video em preto e branco.
Pode ler e escrever videos.

Na leitura:
    - Lê qualquer tipo de video
    - Converte RGB pra bw
    - Salva frmaes individuais

Na escrita:
    - Escreve um arqquivo .mkv *
    - Converte cada frame salvo em um frame real
    - Salva o arquivo

* O tipo mkv é melhor para salvar coisas em bw,
já que o mpeg tem uma compressão mais agressiva.
Foi mudando o tipo de codec que eu achei esse melhor 
(pode ter outro que seja ainda melhor).

As funções de vetorização e expansão lidam
com a conversão de/para a matrix de frames nas colunas.
Usar o "expand" vai apagar todos os frames originais.

Documentação da lib usada (pyav): https://pyav.org/docs/stable/
'''
class VideoBW:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.width = 0
        self.height = 0
        self.fps = 0


    def read(self, path: str):
        # Abrir e carregar video
        container = av.open(path, mode="r")

        # Melhora a velocidade com threads
        container.streams.video[0].thread_type = "AUTO"
        stream = container.streams.video[0]

        # Metadados
        self.width = stream.width
        self.height = stream.height

        rate = (
            stream.average_rate
            or stream.guessed_rate
        )
        
        self.fps = float(rate) if rate else 0.0

        # Ler frames e salvar no array
        for packet in container.demux(stream):
            #print(packet)
            for frame in packet.decode():
                # Converter RGB pra bw
                nframe = frame.to_ndarray(format="rgb24")
                self.frames.append(luminance(nframe))
                    
        
            
    def write(self, path: str):
        container = av.open(path, mode="w")

        stream = container.add_stream("libx264", rate=int(self.fps))
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p"

        for nframe in self.frames:
            frame = av.VideoFrame.from_ndarray(nframe, format="gray")

            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()


    def vectorize(self) -> np.ndarray:
        w = len(self.frames)
        h = self.width * self.height
        
        matrix = np.empty((h, w), dtype=np.uint8)
        for i, frame in enumerate(self.frames):
            matrix[:, i] = frame.flatten()
        
        return matrix


    def expand(self, matrix: np.ndarray):
        # Limpar todos os frames
        self.frames.clear()
        
        for i in range(matrix.shape[1]):
            frame = matrix[:, i].reshape(self.height, self.width)
            self.frames.append(frame)

    
