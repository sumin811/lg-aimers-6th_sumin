import time
from utils import *


def main(config_params, model_params):
    start_time = time.time()  # 시작 시간 기록

    # seed 고정
    set_seed(config_params['seed'])

    # 데이터 셋 읽기
    df_train, df_test, df_submit = read_data(config_params['data_path'])
    

    # 데이터 전처리
    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)

    base_features, cat_features, num_features, removal_features = make_feature_lists(df_train)

    # 범주형 변수 인코딩
    df_train, df_test = encoding_cat_features(df_train, df_test, cat_features, config_params)

    # 결측 처리
    df_train = filling_missing_values(df_train, cat_features, num_features, config_params['model'])
    df_test = filling_missing_values(df_test, cat_features, num_features, config_params['model'])

    # train 데이터 학습
    models = model_kfold(df_train, base_features, cat_features, config_params, model_params)

    # 제출 파일 생성
    make_submission(models, df_test, df_submit, base_features, config_params['model'])

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 총 실행 시간 계산

    minutes, seconds = divmod(elapsed_time, 60)  # 분, 초로 변환
    print(f"총 실행 시간: {int(minutes)}분 {seconds:.2f}초")  # 실행 시간 출력


if __name__ == "__main__":

    config_params = {
        'seed': 777,
        'data_path': "./Data/",
        'k_fold': 5,
        'model': 'LGBM',
        # XGBoost : XGB
        # LightGBM : LGBM
        # CatBoost : CB
        'encoder': 'target' # ordinal / target
    }

    model_param_dict = {
        'XGB': {
            'random_state': config_params['seed'],
            'n_estimators': 2315,
            'learning_rate': 0.005336160307699386,
            'max_depth': 7,
            'subsample': 0.5161168845819335,
            'colsample_bytree': 0.44075797177416176,
            'reg_lambda': 0.2586520827415974,
            'reg_alpha': 8.969816424664486,
            'gamma': 2.99475304385364,
            'objective': 'binary:logistic',
            'verbosity': 0,
        },
        'LGBM': {
            'random_state': config_params['seed'],
            'n_estimators': 2214,
            'learning_rate': 0.010005710429917426,
            'max_depth': 8,
            'subsample': 0.7496669631803915, 
            'colsample_bytree': 0.6058500650325805, 
            'reg_lambda': 0.15967808723677196, 
            'reg_alpha': 10.719550683496601,
            'objective': 'binary',
            'metric': 'auc',
            'class_weight': 'balanced',
            #'early_stopping_rounds': 50,
            'verbose': -1,
        },
        'CB': {
            'random_seed': config_params['seed'],
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'objective': 'Logloss',
            'auto_class_weights': 'Balanced',
            'verbose': 0,
        },
    }

    model_params = model_param_dict[config_params['model']]

    # 디버깅
    if is_debug_mode():
        config_params['k_fold'] = 2
        if config_params['model'] == 'CB':
            model_param_dict[config_params['model']]['iterations'] = 10
        else:
            model_param_dict[config_params['model']]['n_estimators'] = 10

    main(config_params, model_params)