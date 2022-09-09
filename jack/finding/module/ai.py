from jack.finding.iface import finder


class DPRetriever(finder.IR):
    def __init__(self, index):
        store = None
        q_proc = None
        q_model = None
        super(DPRetriever, self).__init__()


class M3Retriever(finder.IR):

    """
    <M>ulti<M>odal<M>ultia-daptive <R>etriever
    """

    def __init__(self, index):
        store = None
        super(M3Retriever, self).__init__()

    def retrieve(self, query_batch):
        pass
