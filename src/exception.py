import sys  #sys imtract with python interpretator and all messge capture by sys module


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()

    file_name=exc_tb.tb_frame.f_code.co_filename

    error_message="Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )

    return error_message





class CustomException(Exception):
    def __init__(self,error_message,error_detail: sys):
        super().__init__(error_message) #inherit from base class
        self.error_message= error_message_detail(
            error_message,error_detail=error_detail  #method take 2 parametr one error message and error detal which come from class error_detail
        )
    
    def __str__(self):
        return self.error_message