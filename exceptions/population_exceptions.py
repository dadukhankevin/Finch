class IndividualGenesNotArrayType(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"The genes in your individuals are not numpy or cupy arrays, this is likely becaus " \
               f"you are using atypical gene_pools like PromptPool.. Refrain from using layers" \
               f"that require genes to be numpy arrays: {self.message}"
