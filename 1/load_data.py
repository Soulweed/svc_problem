import pandas as pd


def main () :
    print  "Start load csv to dataFrame"
    tr_label = pd.read_csv("./DataSet/tj_01_training.csv")
    testset = pd.read_csv("./DataSet/tj_01_test.csv")
    card = pd.read_csv("./DataSet/tj_01_creditcard_card.csv")
    customer = pd.read_csv("./DataSet/tj_01_creditcard_customer.csv")
    transaction = pd.read_csv("./DataSet/tj_01_creditcard_transaction.csv")



    # print label
    # print card

    tr_card_number = tr_label.ix[:, tr_label.columns != 'label']
    set_tmp01 = tr_card_number.merge(card, on='card_no', how='right')
    set_tmp02 = set_tmp01.merge(customer, on='cst_id',how='right')

    set_tmp03 = testset.merge(card, on='card_no', how='right')
    set_tmp04 = set_tmp03.merge(customer, on='cst_id', how='right')
    set_tmp02 = set_tmp02.assign(label= tr_label.get('label'))

    dataset = set_tmp02.drop(['card_no','bill_cyc','pos_dt_x','cst_id','open_dt','exp_dt','main_zip_cd','cr_line_amt','pos_dt_y'], axis =1)

    dataset.to_csv('./DataSet/train.csv')
    set_tmp04.to_csv('./DataSet/test.csv')

if __name__ == '__main__':
    main()
