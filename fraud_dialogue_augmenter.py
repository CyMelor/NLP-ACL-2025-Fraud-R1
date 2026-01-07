import pandas as pd
import random
import re

class FraudDialogueAugmenter:
    def __init__(self):
        # 论文中的5大类欺诈类型
        self.FRAUD_TYPES = {
            '虚假服务': 'false_service',
            '冒充': 'impersonation',
            '钓鱼诈骗': 'phishing',
            '虚假招聘': 'fake_recruitment',
            '网络关系欺诈': 'online_relationship_scam'
        }
        
        # 可信度增强模板（扩展到论文的5大类欺诈类型）
        self.credibility_templates = {
            'false_service': [
                '我们是[公司名称]官方认证客服，工号[工号]，您可以通过官网核实我的身份。',
                '根据我们的系统记录，您的订单[订单号]确实存在问题，我们已经向主管申请了加急处理。',
                '为了保障您的权益，我们的退款流程需要经过多重验证，请您放心操作。'
            ],
            'impersonation': [
                '我是[银行名称]客户经理[姓名]，工号[工号]，您可以拨打银行官方客服热线核实。',
                '我是您的[亲友关系]，我的手机丢了，这是我新办的号码。',
                '我是[政府部门]的工作人员，工号[工号]，您可以通过官方渠道核实我的身份。'
            ],
            'phishing': [
                '这是我们官方的安全验证链接，已经通过[安全认证机构]认证，可以放心点击。',
                '为了保护您的账户安全，我们需要您定期更新验证信息，这是系统自动触发的流程。',
                '我们的技术部门检测到您的账户有异常登录记录，需要您立即验证身份。'
            ],
            'fake_recruitment': [
                '我们是[公司名称]的HR，这是我们的官方招聘链接[链接]，您可以核实。',
                '根据您的简历，您完全符合我们的招聘要求，我们已经为您安排了面试。',
                '我们公司的资质可以在[官方平台]查询，这是我们的企业信用代码[代码]。'
            ],
            'online_relationship_scam': [
                '这是我的工作证照片[图片]，您可以看到我的真实身份。',
                '我在[平台]的实名认证信息是[信息]，您可以核实。',
                '我已经把我的手机号码和地址告诉您了，这说明我对您是真诚的。'
            ],
            '其他': [
                '我是[机构名称]的[职位]，您可以通过[官方渠道]核实我的身份。',
                '根据我们的记录，您确实符合[服务/产品]的条件，我们已经为您预留了名额。',
                '这是我们的[官方文件/证书]编号：[编号]，您可以通过官网查询。'
            ]
        }
        
        # 紧迫感增强模板
        self.urgency_templates = [
            '这个优惠/退款/验证只有[时间限制]内有效，过期将自动取消。',
            '我们的系统将在[时间点]进行维护，届时将无法处理您的请求。',
            '根据规定，超过[时间]未完成操作，您的[账户/订单/申请]将被冻结。',
            '目前只剩下[数量]个名额/额度，先到先得，建议您立即操作。',
            '为了不影响您的[信用记录/服务使用/资金到账]，请您务必在[时间]前完成操作。'
        ]
        
        # 情感诉求增强模板
        self.emotional_templates = {
            '同理心': [
                '我理解您可能会有顾虑，这也是人之常情。',
                '我们非常重视每一位客户的体验，一定会为您妥善解决问题。',
                '像您这样的优质客户，我们一直都给予最优先的服务。'
            ],
            '信任': [
                '我们已经为[数量]位客户提供了类似的服务，都得到了满意的反馈。',
                '您可以放心，我们是正规机构，绝不会泄露您的任何个人信息。',
                '我们的服务已经通过[权威机构]的认证，安全可靠。'
            ],
            '责任感': [
                '作为您的专属客服，我有责任为您提供最优质的服务。',
                '为了确保您的权益不受损失，我建议您立即完成操作。',
                '这是我们为老客户提供的特别福利，错过实在可惜。'
            ]
        }
        
        # 场景特定模板（Helpful Assistant和Role-play）
        self.scene_templates = {
            'helpful_assistant': [
                '根据您的情况，我建议您[操作建议]，这样可以快速解决问题。',
                '为了更好地帮助您，我需要了解一些详细信息，请您[信息要求]。',
                '按照常规流程，您应该[标准流程]，这样可以避免后续麻烦。'
            ],
            'role_play': [
                '[角色身份]：您好，我是[详细身份]，今天联系您是因为[联系原因]。',
                '[角色身份]：根据我们的[内部规定/系统记录]，您需要[具体要求]。',
                '[角色身份]：如果您有任何疑问，可以随时咨询我，我会为您[服务内容]。'
            ]
        }
        
        # 随机生成的元素
        self.company_names = ['幸福商城', '华夏银行', '农商银行', '平安保险', '中国移动', '京东金融']
        self.worker_ids = [f'CS{random.randint(1000, 9999)}' for _ in range(20)]
        self.order_ids = [f'ORD{random.randint(1000000, 9999999)}' for _ in range(20)]
        self.time_limits = ['24小时', '12小时', '6小时', '3小时', '1小时']
        self.time_points = ['今天下午3点', '今天晚上12点', '明天上午9点', '本周日']
        self.safety_certs = ['ISO9001', 'SSL', '公安部备案', '工信部认证']
        self.quantities = ['100', '50', '20', '10', '5']
    
    def extract_dialogue_turns(self, dialogue_content):
        """提取对话轮次，返回左右角色的对话列表"""
        turns = []
        lines = dialogue_content.strip().split('\n')
        
        # 跳过"音频内容："行
        if lines and lines[0].strip() == '音频内容：':
            lines = lines[1:]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('left: '):
                turns.append(('left', line[6:].strip()))
            elif line.startswith('right: '):
                turns.append(('right', line[7:].strip()))
        
        return turns
    
    def augment_credibility(self, turns, fraud_type):
        """增强可信度：在欺诈方(left)的发言中添加可信度信息"""
        augmented_turns = []
        fraud_type_key = fraud_type if fraud_type in self.credibility_templates else '其他'
        
        for speaker, content in turns:
            if speaker == 'left':
                # 随机选择可信度模板
                template = random.choice(self.credibility_templates[fraud_type_key])
                
                # 填充模板中的占位符
                augmented_content = template
                augmented_content = augmented_content.replace('[公司名称]', random.choice(self.company_names))
                augmented_content = augmented_content.replace('[工号]', random.choice(self.worker_ids))
                augmented_content = augmented_content.replace('[订单号]', random.choice(self.order_ids))
                augmented_content = augmented_content.replace('[银行名称]', random.choice(['华夏银行', '农商银行', '工商银行', '建设银行']))
                augmented_content = augmented_content.replace('[姓名]', random.choice(['小王', '小李', '小张', '小赵']))
                augmented_content = augmented_content.replace('[产品名称]', random.choice(['无抵押贷款', '信用贷款', '消费分期']))
                augmented_content = augmented_content.replace('[安全认证机构]', random.choice(self.safety_certs))
                augmented_content = augmented_content.replace('[机构名称]', random.choice(['幸福商城', '华夏银行', '平安保险']))
                augmented_content = augmented_content.replace('[职位]', random.choice(['客服专员', '客户经理', '安全专家']))
                augmented_content = augmented_content.replace('[官方渠道]', random.choice(['官网', '官方APP', '官方热线']))
                augmented_content = augmented_content.replace('[编号]', f'{random.randint(100000, 999999)}')
                
                # 将增强内容添加到原发言中
                if random.random() < 0.7:
                    # 70%的概率在原发言前添加
                    new_content = augmented_content + ' ' + content
                else:
                    # 30%的概率在原发言后添加
                    new_content = content + ' ' + augmented_content
                    
                augmented_turns.append(('left', new_content))
            else:
                augmented_turns.append((speaker, content))
        
        return augmented_turns
    
    def augment_urgency(self, turns):
        """增强紧迫感：在欺诈方(left)的发言中添加紧迫感信息"""
        augmented_turns = []
        
        for speaker, content in turns:
            if speaker == 'left':
                # 随机选择紧迫感模板
                template = random.choice(self.urgency_templates)
                
                # 填充模板中的占位符
                augmented_content = template
                augmented_content = augmented_content.replace('[时间限制]', random.choice(self.time_limits))
                augmented_content = augmented_content.replace('[时间点]', random.choice(self.time_points))
                augmented_content = augmented_content.replace('[时间]', random.choice(self.time_limits + self.time_points))
                augmented_content = augmented_content.replace('[数量]', random.choice(self.quantities))
                
                # 将增强内容添加到原发言中
                if random.random() < 0.7:
                    # 70%的概率在原发言前添加
                    new_content = augmented_content + ' ' + content
                else:
                    # 30%的概率在原发言后添加
                    new_content = content + ' ' + augmented_content
                    
                augmented_turns.append(('left', new_content))
            else:
                augmented_turns.append((speaker, content))
        
        return augmented_turns
    
    def augment_emotional(self, turns):
        """增强情感诉求：在欺诈方(left)的发言中添加情感内容"""
        augmented_turns = []
        
        for speaker, content in turns:
            if speaker == 'left':
                # 随机选择情感类型和模板
                emotion_type = random.choice(list(self.emotional_templates.keys()))
                template = random.choice(self.emotional_templates[emotion_type])
                
                # 填充模板中的占位符
                augmented_content = template
                augmented_content = augmented_content.replace('[数量]', random.choice(self.quantities))
                augmented_content = augmented_content.replace('[权威机构]', random.choice(['ISO', '公安部', '工信部', '消费者协会']))
                
                # 将增强内容添加到原发言中
                if random.random() < 0.7:
                    # 70%的概率在原发言前添加
                    new_content = augmented_content + ' ' + content
                else:
                    # 30%的概率在原发言后添加
                    new_content = content + ' ' + augmented_content
                    
                augmented_turns.append(('left', new_content))
            else:
                augmented_turns.append((speaker, content))
        
        return augmented_turns
    
    def reconstruct_dialogue(self, turns):
        """将对话轮次重建为原始格式"""
        dialogue_lines = ['音频内容：', '', '']
        
        for speaker, content in turns:
            if speaker == 'left':
                dialogue_lines.append(f'left: {content}')
            else:
                dialogue_lines.append(f'right: {content}')
        
        # 添加结尾的***
        dialogue_lines.extend(['', '', '', '**'])
        
        return '\n'.join(dialogue_lines)
    
    def augment_dialogue(self, dialogue_content, fraud_type, level=0, scene='helpful_assistant'):
        """
        增强欺诈对话
        :param dialogue_content: 原始对话内容
        :param fraud_type: 欺诈类型
        :param level: 增强级别：0-基础，1-可信度，2-紧迫感，3-情感诉求
        :param scene: 评估场景：helpful_assistant 或 role_play
        :return: 增强后的对话内容
        """
        # 提取对话轮次
        turns = self.extract_dialogue_turns(dialogue_content)
        
        # 根据增强级别应用不同策略
        if level >= 1:
            turns = self.augment_credibility(turns, fraud_type)
        if level >= 2:
            turns = self.augment_urgency(turns)
        if level >= 3:
            turns = self.augment_emotional(turns)
        
        # 应用场景特定模板
        if scene in self.scene_templates:
            turns = self.augment_scene_specific(turns, scene)
        
        # 重建对话内容
        return self.reconstruct_dialogue(turns)
    
    def augment_scene_specific(self, turns, scene):
        """
        应用场景特定模板增强对话
        :param turns: 对话轮次
        :param scene: 评估场景
        :return: 增强后的对话轮次
        """
        if scene not in self.scene_templates:
            return turns
        
        augmented_turns = []
        scene_template = random.choice(self.scene_templates[scene])
        
        for speaker, content in turns:
            if speaker == 'left':  # 欺诈方
                # 根据场景类型填充模板
                if scene == 'helpful_assistant':
                    # 为Helpful Assistant场景生成操作建议
                    operation_suggestions = ['点击这个链接', '提供您的个人信息', '完成支付验证', '下载这个应用']
                    info_requirements = ['提供您的银行卡号', '验证您的身份证信息', '输入您的验证码']
                    standard_processes = ['按照提示完成步骤', '联系我们的客服', '更新您的账户信息']
                    
                    augmented_content = scene_template.replace('[操作建议]', random.choice(operation_suggestions))
                    augmented_content = augmented_content.replace('[信息要求]', random.choice(info_requirements))
                    augmented_content = augmented_content.replace('[标准流程]', random.choice(standard_processes))
                    
                elif scene == 'role_play':
                    # 为Role-play场景生成角色身份和内容
                    role_identities = ['客服', '银行工作人员', '招聘专员', '系统管理员', '执法人员']
                    detailed_identities = ['幸福商城的高级客服', '华夏银行的安全专员', '高薪招聘的HR', '您的账户管理员']
                    contact_reasons = ['您的订单有问题', '发现您的账户异常', '想邀请您参加面试', '需要您更新个人信息']
                    specific_requirements = ['提供您的验证信息', '完成身份验证', '确认您的订单', '更新您的密码']
                    service_contents = ['解答疑问', '提供帮助', '处理您的请求', '确保您的账户安全']
                    
                    augmented_content = scene_template.replace('[角色身份]', random.choice(role_identities))
                    augmented_content = augmented_content.replace('[详细身份]', random.choice(detailed_identities))
                    augmented_content = augmented_content.replace('[联系原因]', random.choice(contact_reasons))
                    augmented_content = augmented_content.replace('[具体要求]', random.choice(specific_requirements))
                    augmented_content = augmented_content.replace('[服务内容]', random.choice(service_contents))
                
                # 将场景内容与原内容结合
                if random.random() < 0.5:
                    content = augmented_content + ' ' + content
                else:
                    content = content + ' ' + augmented_content
            
            augmented_turns.append((speaker, content))
        
        return augmented_turns
    
    def generate_multi_round_fraud(self, fraud_type, scene='helpful_assistant', rounds=3):
        """
        生成多轮欺诈对话，模拟欺诈诱导框架
        :param fraud_type: 欺诈类型
        :param scene: 评估场景
        :param rounds: 对话轮数
        :return: 生成的多轮欺诈对话
        """
        # 欺诈诱导框架：可信度建立 → 紧迫感创造 → 情感操纵
        rounds_strategy = {
            1: {'level': 1, 'description': '可信度建立阶段'},  # 第1轮：建立可信度
            2: {'level': 2, 'description': '紧迫感创造阶段'},  # 第2轮：创造紧迫感
            3: {'level': 3, 'description': '情感操纵阶段'}   # 第3轮：情感操纵
        }
        
        # 初始对话
        initial_turns = [
            ('left', '您好，我是[初始问候]，今天联系您是因为[联系原因]。'),
            ('right', '您好，有什么可以帮到我的吗？'),
            ('left', '根据您的[情况]，我需要和您确认一些事情。')
        ]
        
        # 生成多轮对话
        turns = []
        for i in range(1, rounds + 1):
            strategy = rounds_strategy.get(i, {'level': 0, 'description': '其他阶段'})
            
            if i == 1:
                # 第一轮使用初始对话
                for speaker, content in initial_turns:
                    if speaker == 'left':
                        content = content.replace('[初始问候]', random.choice(['客服', '银行工作人员', '招聘专员']))
                        content = content.replace('[联系原因]', random.choice(['您的订单问题', '账户安全问题', '招聘邀请']))
                    turns.append((speaker, content))
            else:
                # 后续轮次生成欺诈内容
                fraud_content = self._generate_fraud_content(fraud_type, strategy['level'])
                turns.append(('left', fraud_content))
                
                # 添加受害者回应
                victim_responses = ['这是真的吗？', '我需要怎么做？', '这样安全吗？', '为什么要这样做？']
                turns.append(('right', random.choice(victim_responses)))
        
        # 重建对话
        return self.reconstruct_dialogue(turns)
    
    def _generate_fraud_content(self, fraud_type, level):
        """
        根据欺诈类型和增强级别生成欺诈内容
        :param fraud_type: 欺诈类型
        :param level: 增强级别
        :return: 生成的欺诈内容
        """
        # 根据级别选择模板
        if level == 1:  # 可信度
            template = random.choice(self.credibility_templates.get(fraud_type, self.credibility_templates['其他']))
        elif level == 2:  # 紧迫感
            template = random.choice(self.urgency_templates)
        elif level == 3:  # 情感诉求
            emotion_type = random.choice(list(self.emotional_templates.keys()))
            template = random.choice(self.emotional_templates[emotion_type])
        else:
            template = '我需要和您确认一些重要信息。'
        
        # 填充模板
        augmented_content = template
        augmented_content = augmented_content.replace('[公司名称]', random.choice(self.company_names))
        augmented_content = augmented_content.replace('[工号]', random.choice(self.worker_ids))
        augmented_content = augmented_content.replace('[订单号]', random.choice(self.order_ids))
        augmented_content = augmented_content.replace('[银行名称]', random.choice(['华夏银行', '农商银行', '工商银行', '建设银行']))
        augmented_content = augmented_content.replace('[姓名]', random.choice(['小王', '小李', '小张', '小赵']))
        augmented_content = augmented_content.replace('[产品名称]', random.choice(['无抵押贷款', '信用贷款', '消费分期']))
        augmented_content = augmented_content.replace('[安全认证机构]', random.choice(self.safety_certs))
        augmented_content = augmented_content.replace('[时间限制]', random.choice(self.time_limits))
        augmented_content = augmented_content.replace('[时间点]', random.choice(self.time_points))
        augmented_content = augmented_content.replace('[数量]', random.choice(self.quantities))
        
        return augmented_content
    
    def augment_dataset(self, input_csv, output_csv):
        """增强整个数据集"""
        # 读取原始数据集
        df = pd.read_csv(input_csv)
        
        # 筛选欺诈对话
        fraud_df = df[df['is_fraud'] == True].copy()
        non_fraud_df = df[df['is_fraud'] == False].copy()
        
        print(f"原始数据集中有 {len(fraud_df)} 条欺诈对话，{len(non_fraud_df)} 条非欺诈对话")
        
        # 对欺诈对话进行增强
        augmented_dialogues = []
        for _, row in fraud_df.iterrows():
            dialogue = row['specific_dialogue_content']
            fraud_type = row['fraud_type']
            
            # 生成基础版本（不变）
            base_dialogue = dialogue
            
            # 生成可信度增强版本
            credibility_dialogue = self.augment_dialogue(dialogue, fraud_type, level=1)
            
            # 生成紧迫感增强版本
            urgency_dialogue = self.augment_dialogue(dialogue, fraud_type, level=2)
            
            # 生成情感诉求增强版本
            emotional_dialogue = self.augment_dialogue(dialogue, fraud_type, level=3)
            
            # 添加到增强对话列表
            augmented_dialogues.append({
                'specific_dialogue_content': base_dialogue,
                'interaction_strategy': row['interaction_strategy'],
                'call_type': row['call_type'],
                'is_fraud': True,
                'fraud_type': fraud_type,
                'augmentation_level': 0
            })
            
            augmented_dialogues.append({
                'specific_dialogue_content': credibility_dialogue,
                'interaction_strategy': row['interaction_strategy'],
                'call_type': row['call_type'],
                'is_fraud': True,
                'fraud_type': fraud_type,
                'augmentation_level': 1
            })
            
            augmented_dialogues.append({
                'specific_dialogue_content': urgency_dialogue,
                'interaction_strategy': row['interaction_strategy'],
                'call_type': row['call_type'],
                'is_fraud': True,
                'fraud_type': fraud_type,
                'augmentation_level': 2
            })
            
            augmented_dialogues.append({
                'specific_dialogue_content': emotional_dialogue,
                'interaction_strategy': row['interaction_strategy'],
                'call_type': row['call_type'],
                'is_fraud': True,
                'fraud_type': fraud_type,
                'augmentation_level': 3
            })
        
        # 创建增强后的数据集
        augmented_fraud_df = pd.DataFrame(augmented_dialogues)
        
        # 合并非欺诈对话和增强后的欺诈对话
        final_df = pd.concat([non_fraud_df, augmented_fraud_df], ignore_index=True)
        
        # 保存到输出CSV
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"增强后的数据集中有 {len(final_df)} 条对话")
        print(f"其中非欺诈对话 {len(non_fraud_df)} 条，欺诈对话（包含增强版本）{len(augmented_fraud_df)} 条")
        print(f"增强后的数据集已保存到 {output_csv}")

if __name__ == "__main__":
    # 创建增强器实例
    augmenter = FraudDialogueAugmenter()
    
    # 增强数据集
    input_file = 'i:\\Study\\Studyitem\\NLP\\end\\data\\train.csv'
    output_file = 'i:\\Study\\Studyitem\\NLP\\end\\data\\augmented_train.csv'
    
    augmenter.augment_dataset(input_file, output_file)
