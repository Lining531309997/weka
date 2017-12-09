/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    C45ModelSelection.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.C45.LiNing;

import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for selecting a C4.5-type split for a given dataset.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 10531 $
 */
public class C45ModelSelection extends ModelSelection {

	/** for serialization */
	private static final long serialVersionUID = 3372204862440821989L;

	/** Minimum number of objects in interval. */
	protected final int m_minNoObj;

	/** Use MDL correction? */
	protected final boolean m_useMDLcorrection;

	/** All the training data */
	protected Instances m_allData; //

	/** Do not relocate split point to actual data value */
	protected final boolean m_doNotMakeSplitPointActualValue;

	/**
	 * Initializes the split selection method with the given parameters.
	 * 
	 * @param minNoObj
	 *            minimum number of instances that have to occur in at least two
	 *            subsets induced by split
	 * @param allData
	 *            FULL training dataset (necessary for selection of split
	 *            points).
	 * @param useMDLcorrection
	 *            whether to use MDL adjustement when finding splits on numeric
	 *            attributes
	 * @param doNotMakeSplitPointActualValue
	 *            if true, split point is not relocated by scanning the entire
	 *            dataset for the closest data value
	 */
	public C45ModelSelection(int minNoObj, Instances allData,
			boolean useMDLcorrection, boolean doNotMakeSplitPointActualValue) {
		m_minNoObj = minNoObj;
		m_allData = allData;
		m_useMDLcorrection = useMDLcorrection;
		m_doNotMakeSplitPointActualValue = doNotMakeSplitPointActualValue;
	}

	/**
	 * Sets reference to training data to null.
	 */
	public void cleanup() {

		m_allData = null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	@Override
	public final ClassifierSplitModel selectModel(Instances data) {

		double minResult; 				// 最小的信息增益率
		C45Split[] currentModel; 		// 存放所有未分类属性产生的模型
		C45Split bestModel = null; 		// 目前为止的最好模型
		NoSplit noSplitModel = null; 	// 代表不用分的模型
		double averageInfoGain = 0; 	// 各模型(currentModel)的平均信息增益
		int validModels = 0; 			// 是否存在有效模型
		boolean multiVal = true; 		// 是否具有多值
		Distribution checkDistribution; // 训练数据集的分布
		Attribute attribute; 			// 属性列集合
		double sumOfWeights; 			// 训练数据集的weight的和
		int i; 							// 循环变量

		try {

//			System.out.println("STEP3.2.1.1 ==> 检测数据集个数以及实例是否为同一类");
			// Check if all Instances belong to one class or if not
			// enough Instances to split.
			checkDistribution = new Distribution(data);
			noSplitModel = new NoSplit(checkDistribution);

			// 第一个条件：要求至少有2 *
			// m_minNoObj(默认值2)个样本才可以进行分裂，因为每一个节点至少分出两个节点，每个节点至少有m_minNoObj个样本
			// 第二个条件：检测这个节点上所有样本是否都属于同一类别
			if (Utils.sm(checkDistribution.total(), 2 * m_minNoObj)
					|| Utils.eq(checkDistribution.total(), checkDistribution
							.perClass(checkDistribution.maxClass()))) {
				return noSplitModel;
			}

			// Check if all attributes are nominal and have a
			// lot of values.
			if (m_allData != null) {
//				System.out.println("STEP3.2.1.2 ==> 检测数据集中属性是否为离散型以及属性值是否为多个");
				Enumeration<Attribute> enu = data.enumerateAttributes();
				while (enu.hasMoreElements()) {
					attribute = enu.nextElement();
					// 第一个条件：检测属性是否为连续性属性(数值型)
					// 第二个条件：检测属性的属性值个数是否小于总样本数*0.3
					if ((attribute.isNumeric())
							|| (Utils.sm(attribute.numValues(),
									(0.3 * m_allData.numInstances())))) {
						multiVal = false;
						break;
					}
				}
			}

			currentModel = new C45Split[data.numAttributes()];
			sumOfWeights = data.sumOfWeights();

			// For each attribute.
//			System.out.println("STEP3.2.1.3 ==> START 调用 " + currentModel.getClass().getName() + "中buildClassifier() ==> 计算每个属性信息增益");
			for (i = 0; i < data.numAttributes(); i++) {

				// Apart from class attribute.
				if (i != (data).classIndex()) {

					// Get models for current attribute.
					currentModel[i] = new C45Split(i, m_minNoObj, sumOfWeights,	m_useMDLcorrection);
					// 根据C45Split分裂模型创建分类器
//					System.out.println("---- 计算属性" + data.get(0).attribute(i) + "的信息增益 ----");
					currentModel[i].buildClassifier(data);

					// Check if useful split for current attribute
					// exists and check for enumerated attributes with
					// a lot of values.
					if (currentModel[i].checkModel()) {
						// 计算C45Split分裂模型的信息增益
						if (m_allData != null) {
							if ((data.attribute(i).isNumeric())
									|| (multiVal || Utils.sm(data.attribute(i)
											.numValues(), (0.3 * m_allData
											.numInstances())))) {
								averageInfoGain = averageInfoGain
										+ currentModel[i].infoGain();
								validModels++;
							}
						} else {
							averageInfoGain = averageInfoGain
									+ currentModel[i].infoGain();
							validModels++;
						}
					}
				} else {
					currentModel[i] = null;
				}
			}
//			System.out.println("STEP3.2.1.3 ==> END 返回 " + getClass().getName() + "中selectModel() ==> 计算每个属性信息增益完成");

			// Check if any useful split was found.
			if (validModels == 0) {
				return noSplitModel;
			}
//			System.out.println("STEP3.2.1.4 ==> 根据属性的信息增益计算平均信息增益");
			averageInfoGain = averageInfoGain / validModels;

			// Find "best" attribute to split on.
//			System.out.println("STEP3.2.1.5 ==> 以平均信息增益做基准遍历属性信息增益 ==> 计算最佳分裂属性");
			minResult = 0;
			for (i = 0; i < data.numAttributes(); i++) {
				if ((i != (data).classIndex())
						&& (currentModel[i].checkModel())) {
//					System.out.println(i + " ==> 计算最佳分裂属性 ==> " + (data).classIndex() + " ==> " + (currentModel[i].checkModel()));
					// Use 1E-3 here to get a closer approximation to the
					// original
					// implementation.
					if ((currentModel[i].infoGain() >= (averageInfoGain - 1E-3))
							&& Utils.gr(currentModel[i].gainRatio(), minResult)) {
						bestModel = currentModel[i];
						minResult = currentModel[i].gainRatio();
					}
				}
			}

			// Check if useful split was found.
			if (Utils.eq(minResult, 0)) {
				return noSplitModel;
			}

			// Add all Instances with unknown values for the corresponding
			// attribute to the distribution for the model, so that
			// the complete distribution is stored with the model.
			bestModel.distribution().addInstWithUnknown(data,
					bestModel.attIndex());

			// Set the split point analogue to C45 if attribute numeric.
			if ((m_allData != null) && (!m_doNotMakeSplitPointActualValue)) {
				bestModel.setSplitPoint(m_allData);
			}
			return bestModel;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	@Override
	public final ClassifierSplitModel selectModel(Instances train,
			Instances test) {

		return selectModel(train);
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10531 $");
	}
}
